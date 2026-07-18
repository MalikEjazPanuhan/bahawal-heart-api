# BAHAWAL CARDIOLOGIST HOSPITAL — Heart Risk API

## Files in this folder (drop them all into your GitHub repo)

| File | Purpose |
|---|---|
| `app.py` | The FastAPI server (the model + API) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container config (Render reads this) |
| `render.yaml` | Render.com Blueprint (auto-config) |
| `model_artifacts/` | The trained Random Forest model + metadata |

## How to drop them on GitHub

### Option A — drag & drop (easiest)
1. Open https://github.com/MalikEjazPanuhan/bahawal-heart-api
2. Click **"uploading an existing file"** link (or **Add file → Upload files**)
3. Drag ALL files and folders from this zip into the upload area
   - Make sure `model_artifacts/` keeps its 3 files
4. Click **Commit changes**

### Option B — git push (if you have git locally)
```bash
cd bahawal-heart-api
# copy the files from this zip into this folder
git add .
git commit -m "deploy: FastAPI + model"
git push origin main
```

## What happens after you push

Reply with the repo URL: `https://github.com/MalikEjazPanuhan/bahawal-heart-api`

Then I'll use the Render API to:
1. Create a new Web Service on your Render account
2. Point it at this public repo
3. Render builds the Docker image
4. Deploys in ~3-5 minutes
5. You get a permanent URL like `https://bahawal-heart-api.onrender.com`
6. I'll update the HF static space to call that URL

## What this API serves

- `GET /` — the hospital React app (if you also push `static_frontend/`)
- `GET /health` — liveness check
- `GET /docs` — Swagger UI
- `POST /predict` — 8 model features → risk probability
- `POST /predict-full` — full hospital form JSON → full response

The React app is already deployed at `https://Malik-2025-heart-api.static.hf.space`
and will start working as soon as the API URL is live.
