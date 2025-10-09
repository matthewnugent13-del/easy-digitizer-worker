# Easy Digitizer — FastAPI Worker

This small server takes a PNG upload, creates a preview image and a DST file,
and stores them in S3 at `jobs/<jobId>/preview.png` and `jobs/<jobId>/output.dst`.

> NOTE: Right now the DST is a FAKE placeholder. Replace the TODO section in `main.py`
with your real digitizer code later.

## What you need (environment variables)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (e.g., `us-east-1`)
- `S3_BUCKET` (your exact bucket name)

## How to run locally (optional)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

## Deploy to Render (click path)
1. Push this folder to a new GitHub repo called `easy-digitizer-worker`.
2. Go to https://render.com → **New** → **Web Service**.
3. Select your repo.
4. Environment: **Python**
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Add Environment Variables on Render:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `S3_BUCKET`
8. Click **Deploy**. Copy the Render URL (e.g., `https://easy-digitizer-worker.onrender.com`).
9. In Vercel → Project → Settings → Environment Variables, set `WORKER_URL` to that URL and Redeploy.
```