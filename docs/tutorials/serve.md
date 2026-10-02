# 上线服务

```bash
python -m clefforge serve --host 0.0.0.0 --port 8000
curl -X POST http://127.0.0.1:8000/decide \
  -H 'Content-Type: application/json' \
  -d '{"query":"选择工具","options":["web_search","python_exec"]}'
```
