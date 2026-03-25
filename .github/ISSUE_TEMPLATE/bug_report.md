---
name: Bug report
about: Report a reproducible problem with SecureMatAgent
title: "[BUG] "
labels: bug
assignees: ''
---

## Bug Description

A clear, concise description of what the bug is.

## Steps to Reproduce

1. Start services with `...`
2. Send query `...`
3. Observe `...`

## Expected Behaviour

What you expected to happen.

## Actual Behaviour

What actually happened. Include error messages and stack traces if available.

```
Paste error output here
```

## Environment

| Item | Value |
|---|---|
| OS | e.g. Windows 11 / Ubuntu 22.04 |
| Python version | e.g. 3.10.12 |
| Ollama version | e.g. 0.3.12 (run `ollama --version`) |
| Ollama model | e.g. qwen2.5:7b |
| Docker Desktop version | e.g. 4.30.0 |
| `LOCAL_DEV` setting | true / false |

## Logs

Attach or paste relevant log output. For the app container:

```bash
docker compose logs app --tail 100
```

For local dev (Loguru output):

```
Paste log lines here
```

## Checklist

- [ ] I confirmed Ollama is running: `ollama list` shows the model
- [ ] I confirmed Qdrant is healthy: `docker compose ps` shows `qdrant` as healthy
- [ ] I checked `LOCAL_DEV=true` is set when running outside Docker
- [ ] I ran `python scripts/check_services.py` and it passes
- [ ] I searched existing issues and this has not been reported before

## Additional Context

Any other context, screenshots, or information that might help diagnose the issue.
