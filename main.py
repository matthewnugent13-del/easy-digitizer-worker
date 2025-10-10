import os, io, uuid
from digitizer import make_dst_and_preview
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from digitizer import make_dst_and_preview
import boto3
from botocore.exceptions import ClientError
from PIL import Image

AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your Vercel domain
    allow_methods=["*"],
    allow_headers=["*"],
)

def put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

@app.get("/")
def root():
    return {"ok": True, "service": "easy-digitizer-worker"}

def signed_url(key: str, seconds: int = 300):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=seconds,
    )

@app.post("/jobs")
async def create_job(file: UploadFile = File(...), colors: int = Form(6)):
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="PNG required")
    job_id = str(uuid.uuid4())

    src_bytes = await file.read()
    put_bytes(f"jobs/{job_id}/source.png", src_bytes, "image/png")

    # generate with chosen color count
    dst_bytes, preview_bytes = make_dst_and_preview(src_bytes, int(colors))

    put_bytes(f"jobs/{job_id}/preview.png", preview_bytes, "image/png")
    put_bytes(f"jobs/{job_id}/output.dst", dst_bytes, "application/octet-stream")

    return {"jobId": job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    key = f"jobs/{job_id}/preview.png"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return {"status": "ready", "previewUrl": signed_url(key, 300)}
    except ClientError:
        return {"status": "processing", "previewUrl": None}
