# Hugging Face Docker Deployment

This project is packaged for a demo deployment as a single Docker container. The
container serves the Vite frontend through nginx on port `7860` and proxies API
requests to the FastAPI backend running on `127.0.0.1:8000` inside the same
container.

## Local Docker Run

Build the image from the repository root:

```bash
docker build -t mitra-hf-demo .
```

Run the image locally:

```bash
docker run --rm \
  -p 127.0.0.1:7860:7860 \
  -e LLM_TYPE=openai \
  -e LLM_API_KEY="$LLM_API_KEY" \
  mitra-hf-demo
```

Open the app:

```text
http://127.0.0.1:7860
```

Check backend health:

```bash
curl http://127.0.0.1:7860/api/health
```

Expected health response shape:

```json
{
  "status": "ok",
  "uptime_seconds": 12.345,
  "llm": {
    "provider": "openai",
    "env_configured": true
  }
}
```

If no LLM variables are provided, the app can still start, but LLM-backed
features will not be usable.

## Runtime Environment Variables

Set these as Docker `-e` values locally or as Hugging Face Space secrets and
variables.

| Name | Required | Use |
| --- | --- | --- |
| `LLM_TYPE` | Yes for LLM features | Provider name, for example `openai`, `anthropic`, or `gemini`. |
| `LLM_API_KEY` | Yes for hosted LLMs | API key. Store this as a secret, not a public variable. |
| `LLM_MODEL` | Optional | Overrides the model configured in `config.ini`. |
| `LLM_GATEWAY_URL` | Optional | Custom gateway/base URL. |
| `LLM_CA_BUNDLE` | Optional | Custom CA bundle path for private gateways. |
| `AUTHDB_HOST` | Optional | PostgreSQL host for auth. |
| `AUTHDB_PORT` | Optional | PostgreSQL port for auth. |
| `AUTHDB_NAME` | Optional | PostgreSQL database name for auth. |
| `AUTHDB_USER` | Optional | PostgreSQL username for auth. |
| `AUTHDB_PASSWORD` | Optional | PostgreSQL password for auth. Store as a secret. |

When PostgreSQL values are not supplied, the app falls back to local SQLite
using `auth.db` inside the container.

## Hugging Face Spaces Deployment

1. Create a new Space from the Hugging Face Spaces page.
2. Choose `Docker` as the Space SDK.
3. Use CPU hardware for a free demo. The image is large because it includes ML
   libraries such as Torch, Ray, CatBoost, LightGBM, XGBoost, SHAP, and FAISS.
4. In Space Settings, add runtime variables and secrets:
   - Add `LLM_TYPE` as a variable.
   - Add `LLM_API_KEY` as a secret.
   - Add `LLM_MODEL`, `LLM_GATEWAY_URL`, or `LLM_CA_BUNDLE` only if needed.
   - Add auth database variables only if using PostgreSQL.
5. Make sure the Space repository root contains the project `Dockerfile`.
6. If the Space repository has a `README.md`, include this metadata at the top:

```yaml
---
title: MITRA Demo
sdk: docker
app_port: 7860
---
```

7. Push the deployment branch to the Space repository:

```bash
git remote add hf https://huggingface.co/spaces/<user-or-org>/<space-name>
git push hf deployment:main
```

Hugging Face rebuilds the Space on each push to the Space repository. Build logs
are visible in the Space UI.

## Post-Deploy Checks

After the Space finishes building, open:

```text
https://<user-or-org>-<space-name>.hf.space
```

Check health:

```text
https://<user-or-org>-<space-name>.hf.space/api/health
```

The health endpoint should return `status: ok`. If `llm.env_configured` is
`false`, check the Space secrets and variables.

## Troubleshooting

If the Space starts but the UI cannot call the API, confirm that the Space uses
`app_port: 7860` and that nginx is running inside the container.

If the image build fails while installing Python dependencies, check the build
logs for the exact package. The current requirements are pinned to versions
validated for the Docker demo path.

If the build fails in the frontend step, check `frontend/package.json`. The
Dockerfile resolves frontend dependencies inside the disposable builder layer
without mutating `frontend/package-lock.json`.

If training or evaluation takes too long on free CPU hardware, keep the demo
limits in `config.ini` low: small upload files, few selected models, one HPT
trial, and short SHAP/Ray timeouts.

## References

- Hugging Face Spaces overview: https://huggingface.co/docs/hub/spaces-overview
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
