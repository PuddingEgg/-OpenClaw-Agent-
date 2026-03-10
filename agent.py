from __future__ import annotations

import argparse
from getpass import getpass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse
import webbrowser

from deepseek_client import DeepSeekClient, DeepSeekError
from prompts import build_context_message, build_provider_notes, build_system_prompt
from readme_kb import ReadmeKB
from state_store import SessionState, StateStore


DEFAULT_STATE_PATH = ".install_coach_state.json"
DEFAULT_README_PATH = "OPENCLAW_README.md"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def resolve_settings(project_root: Path) -> tuple[str, str, str]:
    env_values = load_dotenv(project_root / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or env_values.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or env_values.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("DEEPSEEK_MODEL") or env_values.get("DEEPSEEK_MODEL", DEFAULT_MODEL)

    if not api_key:
        api_key = getpass("Paste your DeepSeek API key (input hidden): ").strip()
    if not api_key:
        raise SystemExit("DeepSeek API key is required.")
    return api_key, base_url, model


def render_reply(payload: dict[str, Any]) -> str:
    parts = [payload.get("answer", "").strip()]
    commands = payload.get("commands") or []
    if commands:
        command_block = "\n".join(commands)
        parts.append(f"Run this next:\n{command_block}")
    ask_back = payload.get("ask_user_to_return", "").strip()
    if ask_back:
        parts.append(ask_back)
    return "\n\n".join(part for part in parts if part)


def parse_last_command(user_message: str) -> str:
    lines = [line.strip() for line in user_message.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return lines[-1]


def build_messages(state: SessionState, context_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(state.messages[-8:])
    messages.append({"role": "user", "content": context_message})
    return messages


class CoachService:
    def __init__(
        self,
        client: DeepSeekClient,
        kb: ReadmeKB,
        store: StateStore,
        state: SessionState,
        coach_has_api_key: bool,
    ) -> None:
        self.client = client
        self.kb = kb
        self.store = store
        self.state = state
        self.coach_has_api_key = coach_has_api_key
        self.lock = threading.RLock()

    def get_public_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "platform": self.state.platform,
                "stage": self.state.stage,
                "messages": list(self.state.messages),
            }

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.state = self.store.reset()
            return {
                "platform": self.state.platform,
                "stage": self.state.stage,
                "messages": list(self.state.messages),
            }

    def handle_user_message(self, user_message: str) -> dict[str, Any]:
        with self.lock:
            payload = chat_once(
                self.client,
                self.kb,
                self.state,
                self.coach_has_api_key,
                user_message,
            )
            reply = render_reply(payload)
            self.state.stage = str(payload.get("stage") or self.state.stage)
            self.state.last_command = parse_last_command(
                (payload.get("commands") or [""])[-1] if payload.get("commands") else user_message
            )
            self.state.last_output = user_message
            self.state.add_message("user", user_message)
            self.state.add_message("assistant", reply)
            self.store.save(self.state)
            return {
                "stage": self.state.stage,
                "reply": reply,
                "commands": payload.get("commands") or [],
                "done": bool(payload.get("done")),
                "messages": list(self.state.messages),
            }


def load_web_ui(project_root: Path) -> str:
    ui_path = project_root / "web_ui.html"
    return ui_path.read_text(encoding="utf-8")


def make_handler(service: CoachService, project_root: Path) -> type[BaseHTTPRequestHandler]:
    html = load_web_ui(project_root)

    class CoachHandler(BaseHTTPRequestHandler):
        def _safe_write(self, body: bytes) -> None:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The browser disconnected before reading the response.
                return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self._safe_write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(html)
                return
            if parsed.path == "/api/state":
                self._send_json(service.get_public_state())
                return
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"

            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/chat":
                message = str(body.get("message") or "").strip()
                if not message:
                    self._send_json({"error": "Message is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    payload = service.handle_user_message(message)
                except DeepSeekError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
                    return
                self._send_json(payload)
                return

            if parsed.path == "/api/reset":
                self._send_json(service.reset())
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return CoachHandler


def run_web_server(service: CoachService, project_root: Path, port: int) -> None:
    handler = make_handler(service, project_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print("OpenClaw Install Coach Web")
    print(f"Open {url} in your browser.")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


def run_self_check(readme_path: Path, state_path: Path) -> None:
    kb = ReadmeKB(readme_path)
    store = StateStore(state_path)
    state = store.load()
    sample_chunks = kb.search("install openclaw onboard gateway", state.stage, limit=3)

    print(f"README path: {readme_path}")
    print(f"Chunks loaded: {len(kb.chunks)}")
    print(f"Detected platform: {state.platform}")
    print("Top install chunks:")
    for chunk in sample_chunks:
        print(f"- {chunk.label}")


def chat_once(
    client: DeepSeekClient,
    kb: ReadmeKB,
    state: SessionState,
    coach_has_api_key: bool,
    user_message: str,
) -> dict[str, Any]:
    chunks = kb.search(user_message, state.stage, limit=4)
    context_message = build_context_message(
        platform_name=state.platform,
        stage=state.stage,
        last_command=state.last_command,
        last_output=state.last_output,
        readme_context=kb.format_chunks(chunks),
        provider_notes=build_provider_notes(),
        coach_has_api_key=coach_has_api_key,
        user_message=user_message,
    )
    messages = build_messages(state, context_message)
    return client.chat_json(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal install coach for OpenClaw.")
    parser.add_argument("--readme", default=DEFAULT_README_PATH, help="Path to the OpenClaw knowledge source.")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="Path to the session state file.")
    parser.add_argument("--self-check", action="store_true", help="Verify local setup without calling DeepSeek.")
    parser.add_argument("--reset", action="store_true", help="Reset saved conversation state before starting.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local web UI.")
    args = parser.parse_args()

    project_root = Path.cwd()
    readme_path = (project_root / args.readme).resolve()
    state_path = (project_root / args.state).resolve()

    if not readme_path.exists():
        raise SystemExit(f"README not found: {readme_path}")

    if args.self_check:
        run_self_check(readme_path, state_path)
        return

    store = StateStore(state_path)
    state = store.reset() if args.reset else store.load()
    kb = ReadmeKB(readme_path)
    api_key, base_url, model = resolve_settings(project_root)
    client = DeepSeekClient(api_key=api_key, base_url=base_url, model=model)
    coach_has_api_key = bool(api_key)
    service = CoachService(client, kb, store, state, coach_has_api_key)
    run_web_server(service, project_root, args.port)


if __name__ == "__main__":
    main()
