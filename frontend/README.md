# RAG Center Frontend

独立的知识库上传管理页面，用于创建知识库和验证文档索引结果。

## Run

```bash
npm install
npm run dev
```

开发服务默认运行在 `http://localhost:5173`，`/api` 请求会代理到 `http://127.0.0.1:8000`。

页面会把最近创建的知识库保存在浏览器本地，方便连续添加验证文件；文档上传仍通过现有后端 API 完成。
