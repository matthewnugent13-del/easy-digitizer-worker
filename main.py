import os, io, uuid, traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError

# --- your digitizer ---
# make sure digitizer.py is in the repo root and has: def make_dst_and_preview(image_bytes: bytes, n_colors: int = 6) -> (bytes, bytes)
from digitizer import make_dst_and_preview

# --- AWS setup ---
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", region_name=AWS_REGION)

def put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

def signed_url(key: str, seconds: int = 300):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=int(seconds),
    )

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # tighten later to your vercel domain if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# health
@app.get("/")
def root():
    return {"ok": True, "service": "easy-digitizer-worker"}

# create job: upload PNG + optional colors (1..8), return jobId
@app.post("/jobs")
async def create_job(file: UploadFile = File(...), colors: int = Form(6)):
    try:
        if not file.filename.lower().endswith(".png"):
            raise HTTPException(status_code=400, detail="PNG required")

        job_id = str(uuid.uuid4())

        # read upload and store original
        src_bytes = await file.read()
        put_bytes(f"jobs/{job_id}/source.png", src_bytes, "image/png")

        # IMPORTANT: call digitizer with position args (works even if your function doesn't accept keywords)
        dst_bytes, preview_bytes = make_dst_and_preview(src_bytes, int(colors))

        # save artifacts
        put_bytes(f"jobs/{job_id}/preview.png", preview_bytes, "image/png")
        put_bytes(f"jobs/{job_id}/output.dst", dst_bytes, "application/octet-stream")

        return {"jobId": job_id}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("JOB ERROR:", tb)
        # return error details so you see them in /docs (instead of generic 500)
        return {"error": str(e), "trace": tb}

# status: tell front-end if preview exists; if so, return a signed URL
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        key = f"jobs/{job_id}/preview.png"
        s3.head_object(Bucket=S3_BUCKET, Key=key)   # raises if not found
        ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "300"))
        return {"status": "ready", "previewUrl": signed_url(key, ttl)}
    except ClientError:
        return {"status": "processing"}

# hand back a signed link to the DST (used by your /api/verify-and-link route if you want worker to sign)
@app.get("/download-link/{job_id}")
def download_link(job_id: str):
    try:
        key = f"jobs/{job_id}/output.dst"
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "300"))
        return {"url": signed_url(key, ttl)}
    except ClientError:
        return {"error": "not_ready"}
