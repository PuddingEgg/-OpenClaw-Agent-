# openclwd

一个免费的 `OpenClaw 安装陪练 Agent`。
<img width="1360" height="782" alt="Screenshot 2026-03-10 at 5 13 41 PM" src="https://github.com/user-attachments/assets/04fe4eb5-b39d-47fa-8ba2-a66cb19e4eca" />

如果你也觉得“AI 都还没开口，安装服务先开价”这件事多少带点行为艺术，这个小项目就是来把这段流程免费化、傻瓜化、顺手阴阳怪气化的。

它做的事情很简单：

- 读取本地的 `OPENCLAW_README.md`
- 把安装相关段落切成知识块
- 用 DeepSeek 生成一步一步的人话指导
- 提供一个本地网页聊天界面

一句话总结：

> 别人卖安装服务，我们卖一句话：`python3 agent.py`

## 它不是 OpenClaw

它只是一个“教你把 OpenClaw 装起来”的小教练，不修改 OpenClaw 源码，不接管你的电脑，不收你安装费，只会认真地让你把报错贴回来。

## 为什么会有这个东西

因为有时候开源软件最难的部分不是“功能”，而是：

- 文档很长
- 名词很多
- 新手看完以后开始怀疑是不是自己不配拥有 AI
- 甚至还有人把“帮你安装一下”做成收费项目

于是这个项目的目标就很朴素：

**把“花钱买陪装”的环节，压缩成一个本地免费网页。**

## 特点

- 纯 Python 标准库，没什么离谱依赖
- 本地跑，默认打开 `http://127.0.0.1:8765`
- 支持从 `.env` 读取 DeepSeek 配置，也支持启动时直接粘贴 API Key
- 保存少量会话状态，适合一步一步排障
- 回答风格被限制成“别废话，下一步干什么”

## 文件说明

- `agent.py`：入口，启动本地服务
- `deepseek_client.py`：调用 DeepSeek Chat Completions
- `readme_kb.py`：把 README 切块并做简单检索
- `prompts.py`：限制模型别乱讲
- `state_store.py`：保存会话状态
- `web_ui.html`：本地聊天界面
- `OPENCLAW_README.md`：给 Agent 用的 OpenClaw 安装知识源
- `.env.example`：环境变量示例

## 运行

先准备 Python 3，然后执行：

```bash
python3 agent.py
```

程序会：

1. 读取 `.env` 里的 `DEEPSEEK_API_KEY`
2. 如果没有，就提示你手动粘贴
3. 启动本地网页
4. 自动打开浏览器

默认地址：

```text
http://127.0.0.1:8765
```

## 可选环境变量

复制示例文件：

```bash
cp .env.example .env
```

可配置项：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 自检

这个命令不会调用 API，只检查知识库能不能正常加载：

```bash
python3 agent.py --self-check
```


## 一个很诚实的商业对比

| 方案 | 你得到什么 | 代价 |
| --- | --- | --- |
| 某些“安装服务” | 有人帮你点下一步 | 钱 |
| 这个仓库 | Agent 帮你点脑子里的下一步 | 免费 |

## 免责声明

这个项目不反对劳动收费。

它只是单纯觉得：

**“复制命令 + 打开网页 + 粘贴报错”这套流程，最好先由软件自己免费完成。**
