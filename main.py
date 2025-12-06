import os, io, uuid, traceback, json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError

from digitizer import make_dst_and_preview

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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "service": "easy-digitizer-worker"}


@app.post("/jobs")
async def create_job(file: UploadFile = File(...), colors: int = Form(6)):
    try:
        if not file.filename.lower().endswith(".png"):
            raise HTTPException(status_code=400, detail="PNG required")

        job_id = str(uuid.uuid4())

        src_bytes = await file.read()
        put_bytes(f"jobs/{job_id}/source.png", src_bytes, "image/png")

        # NEW: get palette + trim_count from digitizer
        dst_bytes, preview_bytes, palette, trim_count = make_dst_and_preview(
            src_bytes, int(colors)
        )

        put_bytes(f"jobs/{job_id}/preview.png", preview_bytes, "image/png")
        put_bytes(
            f"jobs/{job_id}/output.dst", dst_bytes, "application/octet-stream"
        )

        meta = {
            "palette": palette,
            "trimCount": int(trim_count),
        }
        put_bytes(
            f"jobs/{job_id}/meta.json",
            json.dumps(meta).encode("utf-8"),
            "application/json",
        )

        return {"jobId": job_id}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("JOB ERROR:", tb)
        return {"error": str(e), "trace": tb}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        preview_key = f"jobs/{job_id}/preview.png"
        s3.head_object(Bucket=S3_BUCKET, Key=preview_key)

        ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "300"))
        preview_url = signed_url(preview_key, ttl)

        meta_key = f"jobs/{job_id}/meta.json"
        palette = None
        trim_count = None

        try:
            meta_obj = s3.get_object(Bucket=S3_BUCKET, Key=meta_key)
            meta_bytes = meta_obj["Body"].read()
            meta = json.loads(meta_bytes.decode("utf-8"))
            palette = meta.get("palette")
            trim_count = meta.get("trimCount")
        except ClientError:
            pass
        except Exception as e:
            print("META ERROR:", e)

        return {
            "status": "ready",
            "previewUrl": preview_url,
            "palette": palette,
            "trimCount": trim_count,
        }

    except ClientError:
        return {"status": "processing"}


@app.get("/download-link/{job_id}")
def download_link(job_id: str):
    try:
        key = f"jobs/{job_id}/output.dst"
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "300"))
        return {"url": signed_url(key, ttl)}
    except ClientError:
        return {"error": "not_ready"}
