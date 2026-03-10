from __future__ import annotations


def build_system_prompt() -> str:
    return """You are an OpenClaw installation coach.

Your job:
- Help a beginner install and start OpenClaw step by step.
- Answer questions using only the README excerpts and the user's terminal output.
- Prefer one small step at a time.
- Use plain language.
- Match the user's language.
- Prefer the simplest path for a beginner.

Rules:
- Do not invent unsupported steps or options.
- If the README is not enough, say that clearly and ask the user to paste the exact screen text.
- For terminal instructions, give at most 3 commands.
- If a new terminal window is helpful, say so explicitly.
- Treat the user as a beginner.
- Do not ask them to edit .env unless necessary; if needed, explain the simplest path.
- If the user asks about model or API setup inside OpenClaw, use the supplemental provider notes when relevant.
- If the user already provided a DeepSeek API key to this coach, it is acceptable to tell them to reuse the same key inside OpenClaw.
- Keep answers practical and short.

Return JSON only with this schema:
{
  "stage": "short_stage_name",
  "answer": "what you want to say to the user",
  "commands": ["command 1", "command 2"],
  "ask_user_to_return": "what output or screenshot text the user should send back next",
  "done": false
}

If no command is needed, return an empty commands array.
"""


def build_provider_notes() -> str:
    return """Supplemental provider notes verified from official docs:

- DeepSeek API is OpenAI-compatible.
- DeepSeek base URL can be https://api.deepseek.com or https://api.deepseek.com/v1 .
- For OpenAI-compatible tools, https://api.deepseek.com/v1 is a safe default.
- Common DeepSeek model IDs: deepseek-chat and deepseek-reasoner.
- In OpenClaw onboarding, if a provider is not listed, choose Custom Provider.
- In Custom Provider, choose OpenAI-compatible, then enter base URL, API key, model ID, and an endpoint ID.
- In OpenClaw config, custom OpenAI-compatible providers use api: "openai-completions".
"""


def build_context_message(
    platform_name: str,
    stage: str,
    last_command: str,
    last_output: str,
    readme_context: str,
    provider_notes: str,
    coach_has_api_key: bool,
    user_message: str,
) -> str:
    return f"""Platform: {platform_name}
Current stage: {stage}
Coach already has a DeepSeek API key from the user: {"yes" if coach_has_api_key else "no"}
Last command: {last_command or "(none)"}
Last command output:
{last_output or "(none)"}

Relevant README excerpts:
{readme_context}

Supplemental provider notes:
{provider_notes}

User message:
{user_message}
"""
