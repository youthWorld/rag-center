# RAG Center Frontend

独立的知识库上传管理页面，用于创建知识库和验证文档索引结果。

## Run

```bash
npm install
npm run dev
```

开发服务默认运行在 `http://localhost:15173`，`/api` 请求会代理到 `http://127.0.0.1:8000`。

知识库通过 URL 中的 `kb_id` 或当前页面刚创建的知识库进入工作区；刷新页面后不会恢复旧的知识库记录，需要从知识库列表重新进入。文档上传仍通过现有后端 API 完成。

## 鉴权配置

复制环境变量示例并填写后端生成的 API Key：

```powershell
Copy-Item .env.example .env
uv run python ../scripts/create_api_key.py --tenant-id tenant_demo --name "本地前端"
```

把命令输出的明文 Key 写入 `frontend/.env` 的 `API_KEY`，然后重启前端开发服务。前端会自动为 API 请求添加 `Authorization: Bearer <key>`。

本地调试时若后端设置 `AUTH_ENABLED=false`，可以将 `API_KEY` 留空，前端不会发送 Authorization 请求头。
