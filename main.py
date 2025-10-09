import os, io, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

def signed_url(key: str, seconds: int = 300):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=seconds,
    )

@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="PNG required")
    job_id = str(uuid.uuid4())

    src_bytes = await file.read()
    # Save the original
    put_bytes(f"jobs/{job_id}/source.png", src_bytes, "image/png")

    # TODO: replace this block with your REAL digitizer:
    # For now we just echo the PNG back as preview and write a fake DST.
    img = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preview_bytes = buf.getvalue()
    dst_bytes = b";FAKE_DST_FOR_TESTING\n"

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
