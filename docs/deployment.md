# 部署

- **Docker**：`deploy/Dockerfile`。
- **Compose**：`deploy/docker-compose.yml`。
- **K8s**：`deploy/k8s/*.yaml`。
- **Helm**：`deploy/helm/`。

服务暴露 `POST /decide`，请求体 `{model, query, options, image}`。
