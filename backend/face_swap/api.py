
#====
"""
Simplified Production Video Face Swap API
Only essential endpoints with S3 integration and threading
"""
 
# =============================================================================
# IMPORTS
# =============================================================================
 
# FastAPI and dependencies
from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, Form, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
 
# Database imports
from sqlalchemy import create_engine, MetaData, text, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
 
# Pydantic models
from pydantic import BaseModel
 
# Standard library imports
import os
import uuid
import json
import shutil
import threading
import asyncio 
import gc
from datetime import datetime, timezone
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import traceback
import requests
import shutil
import cv2
import os
from fastapi import HTTPException
import boto3
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import json
import requests
import asyncio
import cv2
import time
import threading
import onnxruntime as ort
import gc
import numpy as np
from pathlib import Path
from face_swap.resnetface import ResnetFace


import re
import os
import json
import cv2
import tempfile
import numpy as np
from sqlalchemy import text
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from uuid import UUID

  
# Third-party imports
import numpy as np
import boto3
import urllib3
import pytz
from botocore.exceptions import NoCredentialsError
 
# Local imports
from face_swap.face_swap import run_face_swap, get_images_from_group, crop_faces
from config import UPLOAD_FOLDER, TEMPLATES_FOLDER


#check NSFW 
NSFW_CHECK_ENABLED = True

 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
# S3 Configuration
S3_ENDPOINT = "https://sosnm1.shakticloud.ai:9024"
ACCESS_KEY = "immersouser"
SECRET_KEY = "7asiEMavUpJwrWgEUEKGmjMl7NkihTLYLZvjiU/V"
BUCKET_NAME = "immersobuk01"
S3_BASE_URL = f"{S3_ENDPOINT}/{BUCKET_NAME}/VFS/detected-faces"
PRE_SIGNED_URL_EXPIRATION_TIME = 86400  
 
# Database Configuration
# DATABASE_URL = os.getenv("DATABASE_URL",
#     "postgresql://erosuniverse_owner:npg_feSL0XsMKhC5@ep-patient-flower-a8gq4zet.eastus2.azure.neon.tech/erosuniverse?sslmode=require")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", 
    "postgresql://postgres:enePuf77vyIHTZGk@103.114.153.167:49502/postgres?sslmode=require")


# Template Configuration
VIDEO_TEMPLATE_DIR = "./templates"
BASE_VIDEO_TEMPLATE_URL = "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/Templates"
 
# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=8)
 
# Timezone configuration
IST = pytz.timezone('Asia/Kolkata')
 
# Disable SSL certificate verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# =============================================================================
# PYDANTIC MODELS
# =============================================================================
 
class EnhancementSettings(BaseModel):
    smoothness: int
    colorMatch: int
    sharpness: int
 
class FaceSwapRequest(BaseModel):
    group_ids: List[int]
    enhancement_settings: Optional[EnhancementSettings] = None
 
# =============================================================================
# DATABASE SETUP
# =============================================================================
 
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Ensure connections are alive before using
    pool_recycle=300     # Recycle connections every 5 minutes
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData(schema="creator_tools")
Base = declarative_base(metadata=metadata)
 
# Ensure schema exists
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS creator_tools"))
    conn.commit()
 
class VideoFaceSwap(Base):
    __tablename__ = "videofaceswap"
    __table_args__ = {"schema": "creator_tools"}
 
    generation_id = Column(String, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False)
    template_id = Column(String, nullable=False)
    detected_face_url = Column(String, nullable=True)
    target_url = Column(String, nullable=True)
    swap_url = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    finished_at = Column(DateTime, nullable=True)
    credits = Column(Integer, nullable=False, default=50)


 
# Create table if not exists
Base.metadata.create_all(bind=engine)
 
def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        db.execute(text("SET TIME ZONE 'Asia/Kolkata';"))
        yield db
    finally:
        db.close()
 
# =============================================================================
# S3 CLIENT SETUP AND UTILITIES
# =============================================================================
 
def get_s3_client():
    """Create S3 client"""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        verify=False
    )
 
def upload_bytes_to_s3(file_bytes: bytes, s3_key: str, content_type: str = "image/jpeg") -> str:
    """Upload bytes directly to S3"""
    try:
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type
        )
        return f"{S3_ENDPOINT}/{BUCKET_NAME}/{s3_key}"
    except Exception as e:
        print(f"[ERROR] S3 upload failed: {e}")
        raise
 
def upload_file_to_s3(file_path: str, s3_key: str, content_type: str = "video/mp4") -> str:
    """Upload file to S3"""
    try:
        s3_client = get_s3_client()
        with open(file_path, 'rb') as f:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=f,
                ContentType=content_type
            )
        return f"{S3_ENDPOINT}/{BUCKET_NAME}/{s3_key}"
    except Exception as e:
        print(f"[ERROR] S3 file upload failed: {e}")
        raise
 
def generate_signed_url(file_name: str, expiration: int = 86400):
    """Generate signed URL for S3 objects"""
    try:
        s3_client = get_s3_client()
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': f"VFS/Templates/{file_name}"},
            ExpiresIn=expiration
        )
        return signed_url
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="S3 credentials are incorrect.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
 
def get_cors_headers():
    """Get CORS headers"""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Expose-Headers": "Content-Disposition"
    }
 
 
def validate_user_id(db: Session, user_id: str):
    """Check if the user_id exists in the erosuniverse_interface.erosapp_user table"""
    try:
        result = db.execute(
            text("SELECT 1 FROM erosuniverse_interface.erosapp_user WHERE id = :user_id").bindparams(user_id=user_id)
        )
       
        if result.fetchone() is None:
            raise HTTPException(status_code=403, detail="User ID is invalid or unauthorized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking user ID: {str(e)}")
 
 
# =============================================================================
# BACKGROUND TASK FUNCTIONS
# =============================================================================
 
async def upload_detected_faces_to_s3_and_update_db(
    images_data: dict,
    generation_id: str,
    db_session_maker
):
    """Background task to upload detected faces to S3 and update database"""
    try:
        print(f"[DEBUG] Starting S3 upload for generation: {generation_id}")
       
        s3_urls_by_group = {}
       
        # Process each face group
        for group_key, image_files in images_data.items():
            if not image_files:
                continue
               
            group_s3_urls = []
           
            for image_file in image_files:
                try:
                    # Read image file
                    image_path = os.path.join(UPLOAD_FOLDER, generation_id, 'cropped_faces', image_file)
                   
                    if os.path.exists(image_path):
                        with open(image_path, 'rb') as f:
                            image_bytes = f.read()
                       
                        # Upload to S3
                        s3_key = f"VFS/detected-faces/{generation_id}/{group_key}/{image_file}"
                        s3_url = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            upload_bytes_to_s3,
                            image_bytes,
                            s3_key,
                            "image/jpeg"
                        )
                       
                        group_s3_urls.append(s3_url)
                        print(f"[DEBUG] Uploaded {image_file} to S3: {s3_url}")
                       
                except Exception as e:
                    print(f"[ERROR] Failed to upload {image_file}: {e}")
           
            if group_s3_urls:
                s3_urls_by_group[group_key] = group_s3_urls
       
        # Update database
        print(f"[DEBUG] Updating database for generation: {generation_id}")
       
        db = db_session_maker()
        try:
            record = db.query(VideoFaceSwap).filter(
                VideoFaceSwap.generation_id == generation_id
            ).first()
           
            if record:
                record.detected_face_url = json.dumps(s3_urls_by_group)
                db.commit()
                print(f"[DEBUG] Updated database for generation {generation_id}")
               
        finally:
            db.close()
       
        print(f"[DEBUG] S3 upload and DB update completed for generation: {generation_id}")
       
    except Exception as e:
        print(f"[ERROR] Background S3 upload failed: {e}")
 
async def cleanup_local_files(generation_id: str):
    """Cleanup local files after processing"""
    try:
        user_dir = os.path.join(UPLOAD_FOLDER, generation_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            print(f"[DEBUG] Cleaned up local directory: {user_dir}")
        gc.collect()
    except Exception as e:
        print(f"[ERROR] Cleanup failed for {generation_id}: {e}")
 
async def upload_to_s3(result_file_path: str, uid: str) -> str:
    """Upload the result file to S3 and return the URL"""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            verify=False
        )
        s3_key = f"VFS/results/{uid}/result.mp4"
       
        # Upload the file to S3
        s3_client.upload_file(result_file_path, BUCKET_NAME, s3_key)
 
        # Generate the public URL for the uploaded file
        s3_url = f"{S3_ENDPOINT}/{BUCKET_NAME}/{s3_key}"
        return s3_url
    except Exception as e:
        print(f"[ERROR] Failed to upload video to S3: {str(e)}")
        raise RuntimeError(f"S3 Upload failed: {str(e)}")
 
# =============================================================================
# FASTAPI APP SETUP
# =============================================================================
 
app = FastAPI(
    title="Production Video Face Swap API",
    description="Essential video face swap API with S3 integration",
    version="1.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)
 
router = APIRouter()


from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os

router = APIRouter()
 
#new

import threading

# Global Lock for face swap queue
swap_lock = threading.Lock()



# =============================================================================
# API ENDPOINTS
# =============================================================================
 
# Update the S3 upload directory to the specified Thumbnails path
S3_THUMBNAILS_PATH = "VFS/Thumbnails/"
 
def generate_s3_signed_url_for_thumbnail(template_id: str) -> str:
    """Generate signed URL for the thumbnail stored in S3."""
    s3_key = f"VFS/Thumbnails/{template_id}.png"  # Updated to match .png extension
   
    try:
        # Generate a signed URL for the existing thumbnail in S3
        s3_client = get_s3_client()
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=PRE_SIGNED_URL_EXPIRATION_TIME  # URL expiry time in seconds
        )
        return signed_url
    except Exception as e:
        print(f"[ERROR] Failed to generate signed URL for thumbnail: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate signed URL for thumbnail")
 

@router.get("/video-templates")
def list_video_templates_with_thumbnails():
    """List available video templates with thumbnails and signed URLs"""
    try:
        supported_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        video_templates = []
        
        # S3 URL for templates
        s3_base_url = "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/Templates/"
        
        # List files from S3, assuming you have a function to list objects from S3
        s3_client = get_s3_client()
        response = s3_client.list_objects_v2(Bucket="immersobuk01", Prefix="VFS/Templates/")
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail="No templates found in S3.")
        
        for obj in response['Contents']:
            file = obj['Key']
            if file.lower().endswith(supported_extensions):
                template_id = os.path.splitext(os.path.basename(file))[0]
                
                # Get the signed URL for the existing thumbnail in S3
                thumbnail_url = generate_s3_signed_url_for_thumbnail(template_id)
                
                # Generate signed URL for the video file (use s3_client to get a presigned URL)
                signed_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': 'immersobuk01', 'Key': file},
                    ExpiresIn=3600  # URL will expire in 1 hour
                )

                video_templates.append({
                    "template_id": template_id,
                    "filename": os.path.basename(file),
                    "template_url": signed_url,  # Return the signed URL for the video template
                    "thumbnail_url": thumbnail_url  # Return the signed URL for the thumbnail
                })
        
        return {
            "available_video_templates": sorted(video_templates, key=lambda x: x["template_id"]),
            "count": len(video_templates)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/video-templates/{template_id}")
def get_template_info(template_id: str):
    """Get a specific video template's details including signed URL"""
    try:
        s3_client = get_s3_client()
        s3_key = f"VFS/Templates/{template_id}.mp4"
       
        # Check if object exists
        response = s3_client.list_objects_v2(Bucket="immersobuk01", Prefix=s3_key)
        if 'Contents' not in response or len(response['Contents']) == 0:
            raise HTTPException(status_code=404, detail="Template not found in S3.")
       
        # Generate signed URL
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'immersobuk01', 'Key': s3_key},
            ExpiresIn=3600  # 1 hour
        )
       
        # Dummy static metadata, you can replace this with actual metadata reading if required
        response_data = {
            "template_id": template_id,
            "presigned_template_url": signed_url,
            "credits": 40,
            "video_dimensions": {
                "width": 1920,
                "height": 1080,
                "channels": 3
            },
            "swap_type": "video-face-swap"
        }
       
        return JSONResponse(status_code=200, content=response_data)
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 

 
def generate_signed_url_for_image(s3_key: str, expiration: int = 3600):
    """Generate signed URL for images with inline content disposition"""
    try:
        s3_client = get_s3_client()
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key, 'ResponseContentDisposition': 'inline', 'ResponseContentType': 'image/jpeg'},
            ExpiresIn=expiration,
            HttpMethod='GET'
        )
        return signed_url
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="S3 credentials are incorrect.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   


def download_file_from_s3(s3_key: str, destination_path: str):
    s3_client = get_s3_client()
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        s3_client.download_file(BUCKET_NAME, s3_key, destination_path)
    except ClientError:
        raise HTTPException(status_code=404, detail=f"Missing file from S3: {os.path.basename(destination_path)}")

from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from face_swap.face_swap import crop_faces

executor = ThreadPoolExecutor(max_workers=8)

def s3_key_exists(s3_client, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError:
        return False

def upload_local_preprocessing_to_s3(template_id: str):
    """Uploads local preprocessed files to S3 under uploaded_videos/template_id"""
    base_path = os.path.join(UPLOAD_FOLDER, template_id)
    s3_base = f"VFS/uploaded_videos/{template_id}/"

    s3_client = get_s3_client()
    files_to_upload = ['face_embeddings.npy', 'face_bboxes.npy', 'face_kps.npy', 'all_info.json']
    for fname in files_to_upload:
        fpath = os.path.join(base_path, fname)
        if os.path.exists(fpath):
            s3_client.upload_file(fpath, BUCKET_NAME, s3_base + fname)

    cropped_faces_path = os.path.join(base_path, "cropped_faces")
    for root, _, files in os.walk(cropped_faces_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path)
            s3_key = f"{s3_base}{rel_path.replace(os.sep, '/')}"
            s3_client.upload_file(full_path, BUCKET_NAME, s3_key)
            



from urllib.parse import urlparse
import mimetypes
import boto3

S3_ENDPOINT = "https://sosnm1.shakticloud.ai:9024"
ACCESS_KEY = "immersouser"
SECRET_KEY = "7asiEMavUpJwrWgEUEKGmjMl7NkihTLYLZvjiU/V"
BUCKET_NAME = "immersobuk01"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=S3_ENDPOINT,
    region_name="us-east-1"
)

def extract_key_from_urls(url):
    parsed = urlparse(url)
    path = parsed.path
    parts = path.lstrip("/").split("/", 1)
    if parts[0] == BUCKET_NAME:
        return parts[1]
    return path.lstrip("/")

def guess_mime_type(key):
    mime_type, _ = mimetypes.guess_type(key)
    return mime_type or 'application/octet-stream'

def generate_presigned_url(key, expires=31536000):
    return s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': BUCKET_NAME,
            'Key': key,
            'ResponseContentDisposition': 'attachment',  # ✅ force download
            'ResponseContentType': guess_mime_type(key)
        },
        ExpiresIn=expires
    )



@router.post("/uploadvideo/")
async def upload_video(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    video: Optional[UploadFile] = File(None),
    template_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        print(f"[INFO] Processing video upload: user={user_id}, template={template_id}")
 
        # Handle invalid user_id with proper 404 response
        try:
            validate_user_id(db, user_id)
        except Exception:
            return JSONResponse(
                status_code=404,
                content={"code": 404, "error": "User validation failed", "message": "Invalid user_id - user not found"}
            )
 
        generation_id = str(uuid.uuid4())
        user_dir = os.path.join(UPLOAD_FOLDER, generation_id)
        os.makedirs(user_dir, exist_ok=True)
        destination_video_path = os.path.join(user_dir, "input.mp4")
 
        # Initialize thumbnail_url variable
        thumbnail_url = None
        is_uploaded_video = False  # Flag to track if this is an uploaded video
 
        # Either upload video manually or use template_id
        if video is not None and video.filename != "":
            is_uploaded_video = True
            # Process uploaded video file
            filename = video.filename
            file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if file_ext not in ['mp4', 'avi', 'mov', 'mkv']:
                return JSONResponse(
                    status_code=400,
                    content={"code": 400, "error": "Video format not supported", "message": "Only MP4, AVI, MOV, and MKV formats are supported."}
                )
           
            # Save uploaded video
            video_content = await video.read()
            with open(destination_video_path, "wb") as f:
                f.write(video_content)
           
            # Generate thumbnail for uploaded video
            thumbnail_url = await generate_and_upload_thumbnail(destination_video_path, generation_id)
           
            # For uploaded videos, we'll use the generation_id as template_id for processing
            processing_template_id = generation_id
            print(f"[INFO] Using uploaded video file: {filename}")
 
        elif template_id is not None:
            # Download template video from S3
            s3_client = get_s3_client()
            template_video_key = f"VFS/Templates/{template_id}.mp4"
            try:
                s3_client.head_object(Bucket=BUCKET_NAME, Key=template_video_key)
            except ClientError:
                return JSONResponse(
                    status_code=400,
                    content={"code": 400, "error": "Template not found", "message": "Invalid template_id — template video not found in S3"}
                )
           
            s3_client.download_file(BUCKET_NAME, template_video_key, destination_video_path)
            processing_template_id = template_id
            thumbnail_url = generate_s3_signed_url_for_thumbnail(template_id)
            print(f"[INFO] Using template video: {template_id}")
 
        else:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "error": "Missing video input", "message": "No video file or template_id provided."}
            )
 
        # Deduct credits
        deduct_user_credits(db, user_id, amount=2)

        if is_uploaded_video:
            # For uploaded videos, process in main thread
            print(f"[INFO] Starting preprocessing for uploaded video: generation_id={generation_id}")
            
            try:
                print(f"[INFO] Running crop_faces for generation_id: {generation_id}")
                crop_faces(destination_video_path, processing_template_id)
                
                print(f"[INFO] Processing cropped faces for generation_id: {generation_id}")
                detected_faces_urls, face_groups_detected = process_cropped_faces_sync(
                    processing_template_id, user_dir, generation_id
                )
                
                # Check if no faces were detected
                if not face_groups_detected or len(face_groups_detected) == 0:
                    print(f"[WARNING] No faces detected in uploaded video: generation_id={generation_id}")
                    return JSONResponse(
                        status_code=400,
                        content={"code": 400, "error": "Face detection failed", "message": "No faces have been detected in the uploaded video. Please upload a video with visible faces."}
                    )
                
                # Save to DB
                record = VideoFaceSwap(
                    generation_id=generation_id,
                    user_id=user_id,
                    template_id=template_id or generation_id,
                    detected_face_url=json.dumps(detected_faces_urls),
                    target_url="",
                    swap_url="",
                    created_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc)
                )
                db.add(record)
                db.commit()
                db.refresh(record)
                
                return JSONResponse(
                    content={
                        "generation_id": generation_id,
                        "template_id": template_id,
                        "face_groups_detected": face_groups_detected,
                        "detected_faces_urls": detected_faces_urls,
                        "total_face_groups": len(face_groups_detected),
                        "status": "processing",
                        "credits": 50,
                        "thumbnail_url": thumbnail_url
                    },
                    status_code=200
                )
                
            except HTTPException:
                # Re-raise HTTPException (like 400 for no faces detected) without modification
                raise
            except Exception as e:
                print(f"[ERROR] Processing failed for generation_id {generation_id}: {str(e)}")
                
                # Save error record to DB
                record = VideoFaceSwap(
                    generation_id=generation_id,
                    user_id=user_id,
                    template_id=template_id or generation_id,
                    detected_face_url=json.dumps({"error": str(e)}),
                    target_url="",
                    swap_url="",
                    created_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc)
                )
                db.add(record)
                db.commit()
                
                return JSONResponse(
                    status_code=500, 
                    content={"code": 500, "error": "Video processing failed", "message": f"Video preprocessing failed: {str(e)}"}
                )
        
        else:
            # For template videos, process synchronously as before
            s3_client = get_s3_client()
            s3_base = f"VFS/uploaded_videos/{processing_template_id}/"
            required_files = [s3_base + fname for fname in ['face_embeddings.npy', 'face_bboxes.npy', 'face_kps.npy', 'all_info.json']]
            all_exist = all(s3_key_exists(s3_client, key) for key in required_files)

            # Check if cropped_faces exist
            cropped_faces_exist = False
            try:
                resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{s3_base}cropped_faces/")
                cropped_faces_exist = "Contents" in resp and len(resp["Contents"]) > 0
            except Exception:
                cropped_faces_exist = False

            # === If any file missing, run crop_faces and upload ===
            if not all_exist or not cropped_faces_exist:
                print(f"[INFO] Preprocessed data missing. Generating it using crop_faces for template_id={processing_template_id}")
                crop_faces(destination_video_path, processing_template_id)
                upload_local_preprocessing_to_s3(processing_template_id)

            # === Download all required files from S3 ===
            for file_name in ['face_embeddings.npy', 'face_bboxes.npy', 'face_kps.npy', 'all_info.json']:
                s3_key = s3_base + file_name
                destination_path = os.path.join(user_dir, file_name)
                s3_client.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=destination_path)

            # === Process cropped_faces ===
            detected_faces_urls, face_groups_detected = process_cropped_faces_sync(
                processing_template_id, user_dir, generation_id
            )
            
            # Check if no faces were detected for template videos too
            if not face_groups_detected or len(face_groups_detected) == 0:
                print(f"[WARNING] No faces detected in template video: template_id={processing_template_id}")
                return JSONResponse(
                    status_code=400,
                    content={"code": 400, "error": "Face detection failed", "message": "No faces have been detected in the selected template video. Please choose a different template."}
                )
            
            # === Save to DB ===
            record = VideoFaceSwap(
                generation_id=generation_id,
                user_id=user_id,
                template_id=template_id or generation_id,
                detected_face_url=json.dumps(detected_faces_urls),
                target_url="",
                swap_url="",
                created_at=datetime.now(timezone.utc),
                finished_at=None
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            
            return JSONResponse(
                content={
                    "generation_id": generation_id,
                    "template_id": template_id,
                    "face_groups_detected": face_groups_detected,
                    "detected_faces_urls": detected_faces_urls,
                    "total_face_groups": len(face_groups_detected),
                    "status": "processing",
                    "credits": 50,
                    "thumbnail_url": thumbnail_url
                },
                status_code=200
            )
 
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"code": 500, "error": "Server error", "message": str(e)}
        )


def process_cropped_faces_sync(processing_template_id: str, user_dir: str, generation_id: str):
    """
    Process cropped faces synchronously - handles both S3 and local files
    """
    s3_client = get_s3_client()
    cropped_faces_dir = os.path.join(user_dir, "cropped_faces")
    
    # Check if we have local cropped faces directory
    if os.path.exists(cropped_faces_dir) and os.path.isdir(cropped_faces_dir):
        # Process local files
        print(f"[INFO] Processing local cropped faces in: {cropped_faces_dir}")
        group_to_image = {}
        
        # Walk through local directory
        for root, _, files in os.walk(cropped_faces_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, cropped_faces_dir)
                    parts = relative_path.split(os.sep)
                    
                    if len(parts) >= 2:
                        group_name = parts[0]
                        image_name = parts[1]
                        file_size = os.path.getsize(file_path)
                        group_to_image.setdefault(group_name, []).append((file_path, file_size))
    else:
        # Process from S3
        print(f"[INFO] Processing cropped faces from S3 for template: {processing_template_id}")
        s3_base = f"VFS/uploaded_videos/{processing_template_id}/"
        prefix = f"{s3_base}cropped_faces/"
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        group_to_image = {}
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split('/')
                if len(parts) >= 6:
                    group_name = parts[4]
                    image_name = parts[5]
                    group_to_image.setdefault(group_name, []).append((key, obj["Size"]))

    detected_faces_urls = {}
    face_groups_detected = {}

    def handle_group_image(group_name: str, selected_item: str, image_name: str, is_local: bool):
        try:
            if is_local:
                # Handle local file
                local_path = selected_item
                new_s3_key = f"VFS/detected-faces/{generation_id}/cropped_faces/{group_name}/{image_name}"
                
                # Upload to S3
                s3_client.upload_file(local_path, BUCKET_NAME, new_s3_key)
                signed_url = generate_signed_url_for_image(new_s3_key)
                return group_name, signed_url, len(group_to_image[group_name]), None
            else:
                # Handle S3 file
                s3_key = selected_item
                dst_group_dir = os.path.join(user_dir, "cropped_faces", group_name)
                os.makedirs(dst_group_dir, exist_ok=True)
                local_dst = os.path.join(dst_group_dir, image_name)

                # Download from S3
                s3_client.download_file(BUCKET_NAME, s3_key, local_dst)
                
                # Re-upload to new location
                new_s3_key = f"VFS/detected-faces/{generation_id}/cropped_faces/{group_name}/{image_name}"
                s3_client.upload_file(local_dst, BUCKET_NAME, new_s3_key)
                signed_url = generate_signed_url_for_image(new_s3_key)
                return group_name, signed_url, len(group_to_image[group_name]), None
        except Exception as e:
            return group_name, None, 0, str(e)

    futures = []
    for group_name, images in group_to_image.items():
        if not images:
            continue
            
        # Select largest image in the group
        selected_item, _ = max(images, key=lambda x: x[1])
        image_name = os.path.basename(selected_item)
        
        # Determine if we're processing local or S3 files
        is_local = isinstance(selected_item, str) and os.path.exists(selected_item)
        
        futures.append(executor.submit(handle_group_image, group_name, selected_item, image_name, is_local))

    for future in as_completed(futures):
        group_name, signed_url, count, error = future.result()
        if error:
            print(f"[ERROR] Group {group_name} failed: {error}")
            continue
        detected_faces_urls[group_name] = signed_url
        face_groups_detected[group_name] = count

    return detected_faces_urls, face_groups_detected

# Keep all your existing functions unchanged
async def generate_and_upload_thumbnail(video_path: str, generation_id: str) -> str:
    """
    Generate a high-quality thumbnail from video and upload to S3
   
    Args:
        video_path: Path to the video file
        generation_id: Unique identifier for this generation
       
    Returns:
        Signed URL for the uploaded thumbnail
    """
    import cv2
    from PIL import Image
    import io
   
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
       
        if not cap.isOpened():
            raise Exception("Could not open video file")
       
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
       
        # Seek to a frame around 10% of the video or 2 seconds, whichever is smaller
        target_time = min(2.0, total_frames / fps * 0.1)
        target_frame = int(target_time * fps)
       
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
       
        # Read the frame
        ret, frame = cap.read()
        cap.release()
       
        if not ret:
            raise Exception("Could not read frame from video")
       
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Convert to PIL Image for better quality processing
        pil_image = Image.fromarray(frame_rgb)
       
        # Resize to create a high-quality thumbnail (maintain aspect ratio)
        # Target size: 1280x720 for good quality
        target_width = 1280
        target_height = 720
       
        # Calculate dimensions maintaining aspect ratio
        original_width, original_height = pil_image.size
        aspect_ratio = original_width / original_height
       
        if aspect_ratio > target_width / target_height:
            # Wide image
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Tall image
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
       
        # Resize with high-quality resampling
        thumbnail = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
       
        # Save to bytes buffer with high quality
        img_buffer = io.BytesIO()
        thumbnail.save(img_buffer, format='JPEG', quality=95, optimize=True)
        img_buffer.seek(0)
       
        # Upload to S3
        s3_client = get_s3_client()
        thumbnail_key = f"VFS/thumbnails/{generation_id}.jpg"
       
        s3_client.upload_fileobj(
            img_buffer,
            BUCKET_NAME,
            thumbnail_key,
            ExtraArgs={
                'ContentType': 'image/jpeg',
                'CacheControl': 'max-age=31536000'  # Cache for 1 year
            }
        )
       
        # Generate signed URL
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': thumbnail_key},
            ExpiresIn=3600 * 24 * 7  # 7 days
        )
       
        print(f"[INFO] Thumbnail generated and uploaded for generation_id: {generation_id}")
        return signed_url
       
    except Exception as e:
        print(f"[ERROR] Failed to generate thumbnail: {str(e)}")
        # Return None or a default thumbnail URL if generation fails
        return None






# FORCE CPU-ONLY EXECUTION - MUST BE FIRST
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU completely
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'

# NSFW API URL
NSFW_API_URL = "http://nsfw.runai-project-immerso-innnovation-venture-pvt.inferencing.shakticloud.ai/check"


# Global variables for model
global_face_detector = None
model_lock = threading.Lock()

# ============= CLEANUP SERVICE =============
class FolderCleanupService:
    def __init__(self, folder_path, cleanup_interval_minutes=30, max_file_age_minutes=60):
        """
        Initialize the cleanup service
        
        Args:
            folder_path: Path to the folder to clean up
            cleanup_interval_minutes: How often to run cleanup (default: 30 minutes)
            max_file_age_minutes: Maximum age of files to keep (default: 60 minutes)
        """
        self.folder_path = Path(folder_path)
        self.cleanup_interval = cleanup_interval_minutes * 60  # Convert to seconds
        self.max_file_age = max_file_age_minutes * 60  # Convert to seconds
        self.cleanup_thread = None
        self.stop_event = threading.Event()
        self.running = False
        
    def cleanup_folder(self):
        """Clean up old files in the specified folder"""
        try:
            if not self.folder_path.exists():
                print(f"[CLEANUP] Folder {self.folder_path} does not exist")
                return
            
            current_time = time.time()
            files_deleted = 0
            total_size_freed = 0
            
            print(f"[CLEANUP] Starting cleanup of {self.folder_path}")
            
            # Walk through all files in the folder and subfolders
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        # Get file modification time
                        file_mtime = file_path.stat().st_mtime
                        file_age = current_time - file_mtime
                        
                        # Delete if file is older than max_file_age
                        if file_age > self.max_file_age:
                            file_size = file_path.stat().st_size
                            file_path.unlink()  # Delete the file
                            files_deleted += 1
                            total_size_freed += file_size
                            
                            age_minutes = file_age / 60
                            print(f"[CLEANUP] Deleted: {file_path.name} (age: {age_minutes:.1f} minutes)")
                            
                    except (OSError, FileNotFoundError) as e:
                        print(f"[CLEANUP ERROR] Could not delete {file_path}: {e}")
                        continue
            
            # Clean up empty directories
            empty_dirs_removed = 0
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if dir_path != self.folder_path and not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            empty_dirs_removed += 1
                            print(f"[CLEANUP] Removed empty directory: {dir_path}")
                    except OSError:
                        # Directory not empty or permission error
                        continue
            
            # Log cleanup summary
            size_mb = total_size_freed / (1024 * 1024)
            print(f"[CLEANUP] Completed: {files_deleted} files deleted, "
                  f"{empty_dirs_removed} empty dirs removed, "
                  f"{size_mb:.2f} MB freed")
                  
        except Exception as e:
            print(f"[CLEANUP ERROR] Cleanup failed: {e}")
    
    def _cleanup_worker(self):
        """Background worker that runs cleanup periodically"""
        print(f"[CLEANUP] Service started - cleaning {self.folder_path} every {self.cleanup_interval/60:.1f} minutes")
        
        while not self.stop_event.wait(self.cleanup_interval):
            if not self.stop_event.is_set():
                try:
                    self.cleanup_folder()
                except Exception as e:
                    print(f"[CLEANUP ERROR] Worker error: {e}")
        
        print("[CLEANUP] Service stopped")
    
    def start(self):
        """Start the cleanup service"""
        if self.running:
            print("[CLEANUP] Service already running")
            return
        
        self.stop_event.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.running = True
        print("[CLEANUP] Service started")
    
    def stop(self):
        """Stop the cleanup service"""
        if not self.running:
            print("[CLEANUP] Service not running")
            return
        
        print("[CLEANUP] Stopping service...")
        self.stop_event.set()
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        self.running = False
        print("[CLEANUP] Service stopped")

# Global cleanup service
cleanup_service = None

def start_cleanup_service():
    """Start the cleanup service for uploaded_videos folder"""
    global cleanup_service
    
    if cleanup_service is not None:
        print("[CLEANUP] Service already running")
        return
    
    try:
        # Adjust the path to your uploaded_videos folder
        upload_folder = "./uploaded_videos"  # Change this to your actual path
        
        cleanup_service = FolderCleanupService(
            folder_path=upload_folder,
            cleanup_interval_minutes=30,  # Clean every 30 minutes
            max_file_age_minutes=60      # Delete files older than 60 minutes
        )
        cleanup_service.start()
        
        # Run initial cleanup
        cleanup_service.cleanup_folder()
        
        print(f"[CLEANUP] Cleanup service started for folder: {upload_folder}")
        
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to start cleanup service: {e}")

def stop_cleanup_service():
    """Stop the cleanup service"""
    global cleanup_service
    if cleanup_service:
        cleanup_service.stop()
        cleanup_service = None

# ============= END CLEANUP SERVICE =============

def initialize_face_detector():
    """Initialize the face detector model with GUARANTEED CPU-only execution"""
    global global_face_detector
    
    if global_face_detector is not None:
        return global_face_detector
    
    with model_lock:
        # Double-check in case another thread initialized it
        if global_face_detector is not None:
            return global_face_detector
        
        try:
            
            
            # Configure ONNX Runtime for CPU-only execution with optimal settings
            ort.set_default_logger_severity(3)  # Warning level
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4  
            sess_options.inter_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True  
            sess_options.enable_mem_pattern = True
            
            model_file_path = "../backend/face_swap/weights/resnetface.onnx"
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model weights not found at {model_file_path}")
            
            
            cpu_providers = ['CPUExecutionProvider']
            
            
            
            # Method 1: Try with ResnetFace constructor parameters
            try:
                global_face_detector = ResnetFace(
                    model_file=model_file_path, 
                    session_options=sess_options,
                    providers=cpu_providers
                )
                print("[INFO] Method 1: ResnetFace initialized with CPU providers")
            except TypeError:
                # print("[INFO] Method 1 failed, trying Method 2...")
                
                
                original_inference_session = ort.InferenceSession
                
                def force_cpu_inference_session(model_path, **kwargs):
                    # Remove any GPU-related arguments and force CPU
                    kwargs.pop('session_options', None)
                    kwargs.pop('providers', None)
                    # print("[INFO] Forcing CPU-only InferenceSession")
                    return original_inference_session(
                        model_path, 
                        sess_options=sess_options,
                        providers=cpu_providers
                    )
                
                # Apply the patch
                ort.InferenceSession = force_cpu_inference_session
                try:
                    global_face_detector = ResnetFace(model_file=model_file_path)
                    print("[INFO] Method 2: ResnetFace initialized with patched GPU-only session")
                finally:
                    # Restore original function
                    ort.InferenceSession = original_inference_session
            
            # Test the model to ensure it works
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            start_test = time.time()
            det, kpss = global_face_detector.detect(dummy_img, input_size=(160, 160))
            test_time = time.time() - start_test
            
            
            
            return global_face_detector
            
        except Exception as e:
            global_face_detector = None
            raise


@router.post('/uploadtargetface/{uid}')
async def upload_target_face(
    uid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    perform_nsfw_check: Optional[bool] = NSFW_CHECK_ENABLED
):
    """
    Upload a target face image with GUARANTEED CPU-only face detection
    """
    temp_path = None
    
    try:
        print(f"[INFO] Uploading target face for generation={uid}")

        # Check if file is empty
        if file is None or len(await file.read()) == 0:
            return JSONResponse(
                content={"code": 400, "error": "No image provided", "message": "No image uploaded"},
                status_code=400
            )
        await file.seek(0)
        file_content = await file.read()

        # ---------- NSFW Check ----------
        if perform_nsfw_check:
            try:
                filename = os.path.basename(file.filename)
                temp_nsfw_path = os.path.join(UPLOAD_FOLDER, f"nsfw_{filename}")
                with open(temp_nsfw_path, "wb") as f:
                    f.write(file_content)
                with open(temp_nsfw_path, 'rb') as img_f:
                    res = requests.post(
                        NSFW_API_URL,
                        files={"images": (filename, img_f, 'image/jpeg')},
                        data={"check_animals": "true", "check_nsfw": "true", "check_up_image": "true"}
                    )
                    res.raise_for_status()
                    results = res.json().get("results", [])
                os.remove(temp_nsfw_path)

                if not results or not all(r.get("result", False) for r in results):
                    return JSONResponse(
                        content={"code": 400, "error": "Content blocked", "message": "**Blocked by moderation**\nThis content has been blocked for violating our guidelines."},
                        status_code=400
                    )
            except Exception as e:
                print(f"[NSFW ERROR] {e}")
                if 'temp_nsfw_path' in locals() and os.path.exists(temp_nsfw_path):
                    os.remove(temp_nsfw_path)
                return JSONResponse(
                    content={"code": 400, "error": "Content blocked", "message": "**Blocked by moderation**\nThis content has been blocked for violating our guidelines."},
                    status_code=400
                )

        # ---------- Validate generation_id ----------
        record = db.query(VideoFaceSwap).filter(VideoFaceSwap.generation_id == uid).first()
        if not record:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "error": "Invalid generation ID", "message": "Invalid generation ID"}
            )

        # ---------- Get the global face detector (GUARANTEED CPU-only) ----------
        try:
            face_detector = initialize_face_detector()
        except Exception as e:
            print(f"[ERROR] Failed to get face detector: {e}")
            return JSONResponse(
                status_code=500,
                content={"code": 500, "error": "Service unavailable", "message": "Face detection service unavailable"}
            )

        # ---------- Load and preprocess image ----------
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{uid}.jpg")
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Clear file_content from memory
        del file_content
        gc.collect()
        
        img = cv2.imread(temp_path)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "error": "Image loading failed", "message": "Failed to load image for face detection"}
            )
        
        # Reasonable resizing for CPU processing
        original_height, original_width = img.shape[:2]
        max_dimension = 800  # Larger is fine for CPU
        if max(original_height, original_width) > max_dimension:
            if original_width > original_height:
                new_width = max_dimension
                new_height = int((max_dimension * original_height) / original_width)
            else:
                new_height = max_dimension
                new_width = int((max_dimension * original_width) / original_height)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"[INFO] Resized image from {original_width}x{original_height} to {new_width}x{new_height}")

        # ---------- GUARANTEED CPU-only face detection ----------
        try:
            start_time = time.time()
            
            # Primary detection attempt
            det, kpss = face_detector.detect(img, input_size=(160, 160))
            
            detection_time = time.time() - start_time
            
        except Exception as e:
            print(f"[DETECTION ERROR] Primary attempt failed: {e}")
            
            # Fallback with smaller input size
            try:
                print("[INFO] Retrying with smaller input size (128x128)...")
                det, kpss = face_detector.detect(img, input_size=(128, 128))
                print("[INFO] Fallback detection successful")
            except Exception as e2:
                print(f"[DETECTION ERROR] Fallback also failed: {e2}")
                
                # Final fallback with minimal size
                try:
                    print("[INFO] Final retry with minimal input size (96x96)...")
                    det, kpss = face_detector.detect(img, input_size=(96, 96))
                    print("[INFO] Minimal size detection successful")
                except Exception as e3:
                    print(f"[DETECTION CRITICAL ERROR] All detection attempts failed: {e3}")
                    return JSONResponse(
                        status_code=500,
                        content={"code": 500, "error": "Face detection failed", "message": "Face detection failed - all fallback methods exhausted"}
                    )

        if len(det) == 0:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "error": "No faces detected", "message": "No faces detected in the uploaded image"}
            )

        # ---------- Process detected faces ----------
        uploaded_face_urls = []
        for idx, bbox in enumerate(det):
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Enlarge bounding box to include hair
            border_top = int((y2 - y1) * 0.25)
            border_bottom = int((y2 - y1) * 0.25)
            border_left = int((x2 - x1) * 0.1)
            border_right = int((x2 - x1) * 0.1)

            y1 = max(y1 - border_top, 0)
            y2 = min(y2 + border_bottom, img.shape[0])
            x1 = max(x1 - border_left, 0)
            x2 = min(x2 + border_right, img.shape[1])

            face_img = img[y1:y2, x1:x2]

            # Save cropped face
            timestamp = int(time.time())
            face_filename = f"target_face_{uid}_face_{idx}_{timestamp}.jpg"
            face_path = os.path.join(UPLOAD_FOLDER, face_filename)
            cv2.imwrite(face_path, face_img)

            # Upload to S3
            s3_key = f"VFS/target-faces/{uid}/{face_filename}"
            unsigned_s3_url = f"https://sosnm1.shakticloud.ai:9024/immersobuk01/{s3_key}"

            def upload_to_s3():
                with open(face_path, 'rb') as f:
                    return upload_bytes_to_s3(f.read(), s3_key, "image/jpeg")

            s3_stored_url = await asyncio.get_event_loop().run_in_executor(executor, upload_to_s3)
            uploaded_face_urls.append(unsigned_s3_url)

            # Clean up face file immediately
            if os.path.exists(face_path):
                os.remove(face_path)

        # ---------- Generate signed URLs ----------
        signed_face_urls = []
        for url in uploaded_face_urls:
            s3_key = url.split(f"{S3_ENDPOINT}/{BUCKET_NAME}/")[-1]
            signed_url = get_s3_client().generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
                ExpiresIn=PRE_SIGNED_URL_EXPIRATION_TIME
            )
            signed_face_urls.append(signed_url)

        # ---------- Update DB ----------
        record.target_url = json.dumps({"faces": uploaded_face_urls})
        db.commit()
        db.refresh(record)

        return JSONResponse(
            content={
                "message": f"Successfully uploaded {len(uploaded_face_urls)} faces",
                "generation_id": uid,
                "target_face_urls": uploaded_face_urls,
                "signed_target_face_urls": signed_face_urls,
                "status": "ready_for_swap"
            },
            status_code=200,
            headers=get_cors_headers()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Target face upload failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "error": "Internal server error", "message": "Internal Server Error"}
        )
    
    finally:
        # Comprehensive cleanup
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'img' in locals():
                del img
            gc.collect()
        except Exception as cleanup_error:
            print(f"[WARNING] Cleanup error: {cleanup_error}")

# Modified startup function to include cleanup service
def initialize_app():
    """Call this when your app starts - includes both face detector and cleanup service"""
    try:
        print("[INFO] Initializing application with CPU-only services...")
        
        # Verify CPU-only environment
        print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"[INFO] Available ONNX providers: {ort.get_available_providers()}")
        
        # Initialize face detection
        initialize_face_detector()
        
        # Start cleanup service
        start_cleanup_service()
        
        print("[INFO] All services initialized successfully (CPU-only mode)")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        raise






# NSFW API URL
NSFW_API_URL = "http://nsfw.runai-project-immerso-innnovation-venture-pvt.inferencing.shakticloud.ai/check"


import requests as http_requests  # to avoid confusion with 'requests' used in NSFW API
from fastapi import Form  # for form field



@router.post('/uploadnewfaces/{uid}/{group_id}')
async def upload_new_faces(
    uid: str,
    group_id: str,
    background_tasks: BackgroundTasks,
    image_url: str = Form(...),  # Expect signed URL as form data
    db: Session = Depends(get_db),
    perform_nsfw_check: Optional[bool] = NSFW_CHECK_ENABLED
):
    """Upload new face via signed URL (with optional NSFW validation)"""
    try:
        print(f"[INFO] Uploading new face via URL: generation={uid}, group={group_id}, NSFW check={perform_nsfw_check}")

        # === STEP 1: Download image from signed URL ===
        try:
            response = http_requests.get(image_url, timeout=10)
            response.raise_for_status()
            file_content = response.content

            # Validate it's actually an image (basic check)
            if not response.headers.get("content-type", "").startswith("image/"):
                return JSONResponse(
                    content={"code": 400, "error": "Invalid image URL", "message": "Provided URL does not point to a valid image."},
                    status_code=400
                )
        except Exception as e:
            print(f"[ERROR] Failed to download image from URL: {str(e)}")
            return JSONResponse(
                content={"code": 400, "error": "Image download failed", "message": "Failed to download image from provided URL."},
                status_code=400
            )

        # At this point, we have `file_content` as bytes — same as `await file.read()`
        # Now proceed exactly as before, just skip reading from UploadFile

        if perform_nsfw_check:
            # === STEP 2: NSFW VALIDATION ===
            try:
                # Use a temporary filename for validation
                filename = f"{uid}_{group_id}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                # Save locally for NSFW check
                with open(filepath, "wb") as buffer:
                    buffer.write(file_content)

                with open(filepath, 'rb') as img_file:
                    api_response = http_requests.post(
                        NSFW_API_URL,
                        files={"images": (filename, img_file, 'image/jpeg')},
                        data={"check_animals": "true", "check_nsfw": "true", "check_up_image": "true"}
                    )
                    api_response.raise_for_status()
                    api_json = api_response.json()

                results = api_json.get("results", [])
                if not results or not all(r.get("result", False) for r in results):
                    os.remove(filepath)
                    return JSONResponse(content={
                        "code": 400, 
                        "error": "Content blocked", 
                        "message": "**Blocked by moderation**\nThis content has been blocked for violating our guidelines. You can review the policy or contact support for more details"
                    }, status_code=400)

                # Clean up temporary file
                os.remove(filepath)

            except Exception as e:
                print(f"[ERROR] NSFW check failed: {str(e)}")
                return JSONResponse(content={
                    "code": 400, 
                    "error": "Content blocked", 
                    "message": "**Blocked by moderation**\nThis content has been blocked for violating our guidelines. You can review the policy or contact support for more details"
                }, status_code=400)

        # ✅ Check if generation_id is valid
        record = db.query(VideoFaceSwap).filter(VideoFaceSwap.generation_id == uid).first()
        if not record:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "error": "Invalid generation ID",
                    "message": "Invalid user input"
                }
            )

        # ✅ Validate group_id using all_info.json
        all_info_path = os.path.join(UPLOAD_FOLDER, uid, 'all_info.json')
        if not os.path.exists(all_info_path):
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "error": "Missing face info",
                    "message": "Missing face info file to validate group ID"
                }
            )

        with open(all_info_path, 'r') as f:
            all_info = json.load(f)

        max_groups = int(all_info.get("max_groups", 0))
        if int(group_id) >= max_groups:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "error": "Invalid group ID",
                    "message": f"Invalid group ID {group_id}. Only 0 to {max_groups - 1} are allowed."
                }
            )

        # Save to local new_faces directory
        new_faces_dir = os.path.join(UPLOAD_FOLDER, uid, 'new_faces')
        os.makedirs(new_faces_dir, exist_ok=True)
        file_location = os.path.join(new_faces_dir, f"{group_id}.jpg")

        def save_file_thread():
            with open(file_location, "wb") as file_object:
                file_object.write(file_content)
            print(f"[THREAD] Saved file locally: {file_location}")

        await asyncio.get_event_loop().run_in_executor(executor, save_file_thread)

        # Upload to S3
        s3_file_path = f"VFS/uploads/{uid}/new_faces/{group_id}.jpg"
        def upload_to_s3_thread():
            print(f"[THREAD] Uploading to S3: {s3_file_path}")
            return upload_bytes_to_s3(file_content, s3_file_path, "image/jpeg")

        s3_url = await asyncio.get_event_loop().run_in_executor(executor, upload_to_s3_thread)

        # Generate signed URL for preview
        s3_presigned_url = get_s3_client().generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_file_path},
            ExpiresIn=PRE_SIGNED_URL_EXPIRATION_TIME
        )

    

        return JSONResponse(
            content={
                "message": "Face uploaded successfully",
                "generation_id": uid,
                "group_id": group_id,
                "target_url": s3_presigned_url,
                "status": "ready_for_swap"
            },
            status_code=200,
            headers=get_cors_headers()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Face upload failed: {str(e)}")
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"code": 500, "error": "Internal server error", "message": f"{str(e)}\n{traceback_str}"}
        )




 
def deduct_user_credits(db: Session, user_id: str, amount: int = 2):
    result = db.execute(
        text("""
            SELECT balance FROM erosuniverse_interface.erosapp_usercredits
            WHERE user_id = :user_id
        """).bindparams(user_id=user_id)
    ).fetchone()
 
    if not result:
        raise HTTPException(status_code=404, detail="User credit balance not found")
 
    balance = result[0]
    if balance < amount:
        raise HTTPException(status_code=402, detail="Insufficient credits")
 
    db.execute(
        text("""
            UPDATE erosuniverse_interface.erosapp_usercredits
            SET balance = balance - :amount
            WHERE user_id = :user_id
        """).bindparams(user_id=user_id, amount=amount)
    )
    db.commit()


def cleanup_uploaded_files(uid: str) -> None:
    """
    Clean up all locally saved files for the given `uid` inside `uploaded_videos` directory.
   
    This will delete:
    - cropped faces
    - face embeddings
    - video inputs/outputs
    - any temporary files created during processing.
   
    :param uid: The user ID or generation ID to clean up the files.
    """
    try:
        # Path to the user's uploaded video directory
        user_dir = os.path.join(UPLOAD_FOLDER, uid)
 
        # Check if the directory exists
        if not os.path.exists(user_dir):
            print(f"[INFO] No files found for {uid} to clean.")
            return
 
        # List of subdirectories and files to delete (now handled by deleting the user_dir itself)
        directories_to_delete = []
 
        # Deleting all files and directories
        for item in directories_to_delete:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)  # Delete the directory
                    print(f"[DEBUG] Deleted directory: {item}")
                else:
                    os.remove(item)  # Delete the file
                    print(f"[DEBUG] Deleted file: {item}")
            else:
                print(f"[INFO] File or directory not found: {item}")
        
        # Delete the user directory itself after cleanup
        if os.path.isdir(user_dir):
            try:
                shutil.rmtree(user_dir)
                print(f"[DEBUG] Deleted user directory: {user_dir}")
                print(f"[INFO] Cleanup completed for {user_dir}.")
            except Exception as e:
                print(f"[ERROR] Failed to delete user directory {user_dir}: {e}")
        
        print(f"[INFO] Cleanup completed for {uid}.")
   
    except Exception as e:
        print(f"[ERROR] Failed to clean up files for {uid}: {str(e)}")
 




import os
import json
import threading
import asyncio
from datetime import datetime
from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import subprocess

from face_swap.face_swap import run_face_swap, faceswap_executor

# GPU monitoring with pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    
    # Check if GPU is actually available
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        GPU_AVAILABLE = True
        print("[INFO] GPU detected and available for face swap processing")
    except Exception as e:
        GPU_AVAILABLE = False
        print(f"[INFO] GPU not available, will fallback to CPU processing: {e}")
        
except Exception as e:
    print(f"[WARNING] NVML not available for GPU monitoring: {e}")
    NVML_AVAILABLE = False
    GPU_AVAILABLE = False
    print("[INFO] No GPU monitoring available, will use CPU processing")

# Global lock and queue
swap_lock = threading.Lock()
queue = []

def get_gpu_utilization() -> float:
    """Returns current GPU utilization percentage if available, else 0."""
    if not NVML_AVAILABLE or not GPU_AVAILABLE:
        return 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assuming single GPU
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu  # in percentage
    except Exception as e:
        print(f"[ERROR] Failed to get GPU utilization: {e}")
        return 0.0


async def process_face_swap_threadsafe(
    uid: str,
    group_ids: List[str],
    enhancement_settings,
    loaded_dict: dict,
    db_session_maker: sessionmaker
):
    """Thread-safe wrapper: only use queue/lock if GPU is >=80% utilized"""
    
    # Check if GPU is available
    if not GPU_AVAILABLE:
        print(f"[FALLBACK] No GPU available for {uid} - processing with CPU")
        queue.append(uid)
        try:
            await process_face_swap_simple(
                uid,
                group_ids,
                enhancement_settings,
                loaded_dict,
                db_session_maker
            )
        except Exception as e:
            print(f"[ERROR] Exception during CPU face swap for {uid}: {str(e)}")
            raise
        finally:
            if uid in queue:
                queue.remove(uid)
            if len(queue) > 0:
                print(f"[QUEUE] Queue now: {queue[0]} is next.")
            else:
                print("[QUEUE] Queue is now empty.")
        return

    # Original GPU logic
    gpu_usage = get_gpu_utilization()
    should_queue = gpu_usage >= 80.0

    # Always add to queue for tracking
    queue.append(uid)
    try:
        print(f"[QUEUE] GPU utilization is {gpu_usage:.1f}% -> queueing {'enabled' if should_queue else 'disabled'} for {uid}")

        if len(queue) > 1:
            prev_user = queue[-2] if len(queue) > 1 else None
            if prev_user:
                print(f"[QUEUE] You are in the queue, waiting for {prev_user} to finish (if applicable).")

        if should_queue:
            print(f"[QUEUE] High GPU load ({gpu_usage:.1f}%) - waiting for lock to process {uid}")
            with swap_lock:
                print(f"[QUEUE] Acquired lock for {uid} (GPU-heavy mode)")
                try:
                    await process_face_swap_simple(
                        uid,
                        group_ids,
                        enhancement_settings,
                        loaded_dict,
                        db_session_maker
                    )
                finally:
                    print(f"[QUEUE] Released lock for {uid}")
        else:
            print(f"[QUEUE] Low GPU load ({gpu_usage:.1f}%) - skipping lock, processing {uid} concurrently")
            await process_face_swap_simple(
                uid,
                group_ids,
                enhancement_settings,
                loaded_dict,
                db_session_maker
            )
    except Exception as e:
        print(f"[ERROR] Exception during face swap for {uid}: {str(e)}")
        raise
    finally:
        # Always remove from queue
        if uid in queue:
            queue.remove(uid)
        if len(queue) > 0:
            print(f"[QUEUE] Queue now: {queue[0]} is next.")
        else:
            print("[QUEUE] Queue is now empty.")


# Dummy placeholder for actual processing function
# Replace this with your real `process_face_swap_simple` async function
async def process_face_swap_simple(
    uid: str,
    group_ids: List[str],
    enhancement_settings,
    loaded_dict: dict,
    db_session_maker: sessionmaker
):
    processing_mode = "CPU" if not GPU_AVAILABLE else "GPU"
    print(f"[PROCESS] Starting face swap for {uid} using {processing_mode}...")
    # Simulate async work (replace with real logic)
    await asyncio.sleep(5)
    print(f"[PROCESS] Finished face swap for {uid} using {processing_mode}")




@router.post('/faceswap/{uid}')
async def face_swap(
    uid: str,
    request: FaceSwapRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start face swap processing and return immediately"""
    try:
        group_ids = request.group_ids
        enhancement_settings = request.enhancement_settings

        print(f"[INFO] Starting face swap: generation={uid}, groups={group_ids}")

        # Load face info
        try:
            with open(os.path.join(UPLOAD_FOLDER, uid, 'all_info.json'), 'r') as file:
                loaded_dict = json.load(file)
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"code": 404, "error": "Face info not found", "message": "Face info file not found"}
            )

        # Validate group IDs
        for group_id in group_ids:
            if int(group_id) >= int(loaded_dict['max_groups']):
                return JSONResponse(
                    status_code=400,
                    content={"code": 400, "error": "Invalid group ID", "message": f"Group ID {group_id} not found"}
                )

        # Validate generation ID
        record = db.query(VideoFaceSwap).filter(VideoFaceSwap.generation_id == uid).first()
        if not record:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "error": "Invalid generation ID", 
                    "message": "Invalid generation ID"
                }
            )

        # Check if already completed
        swap_url = getattr(record, "swap_url", None)
        finished_at = getattr(record, "finished_at", None)
        if swap_url is not None and isinstance(swap_url, str) and swap_url != "" and finished_at is not None and isinstance(finished_at, datetime):
            s3_key = swap_url.replace(f"{S3_ENDPOINT}/{BUCKET_NAME}/", "")
            s3_presigned_url = generate_signed_url_for_video(s3_key)
            return JSONResponse(
                content={
                    "status": "completed",
                    "generation_id": uid,
                    "video_url": s3_presigned_url,
                    "message": "Face swap already completed"
                },
                status_code=200
            )

        # Deduct credits
        if record.user_id:
            deduct_user_credits(db, record.user_id, amount=2)

        # ✅ Check that all required new face images exist
        for group_id in group_ids:
            face_file_path = os.path.join(UPLOAD_FOLDER, uid, 'new_faces', f"{group_id}.jpg")
            if not os.path.exists(face_file_path):
                return JSONResponse(
                    status_code=400,
                    content={"code": 400, "error": "Missing face image", "message": f"Missing new face image for group ID {group_id}. Please upload it using /uploadnewfaces/{uid}/{group_id}"}
                )

        # Start background processing
        background_tasks.add_task(
            process_face_swap_threadsafe,
            uid,
            group_ids,
            enhancement_settings,
            loaded_dict,
            SessionLocal
        )

        # Update response message based on processing mode
        processing_mode = "CPU" if not GPU_AVAILABLE else "GPU"
        estimated_time = "3-5 minutes" if not GPU_AVAILABLE else "1-2 minutes"

        # Return immediately
        return JSONResponse(
            content={
                "status": "processing",
                "generation_id": uid,
                "processing_mode": processing_mode,
                "message": f"Face swap processing started using {processing_mode}. Check status using /faceswap/status/{uid}",
                "estimated_time": estimated_time
            },
            status_code=202  # 202 Accepted - request accepted for processing
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Face swap failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "error": "Internal server error", "message": str(e)}
        )


    

import subprocess

LOGO_PATH = "logo.png"

import tempfile

def overlay_logo_on_video(video_path: str, logo_path: str):
    """
    Overlays a transparent logo inside the video frame (not in black margin),
    top-right corner with padding. Scales logo to 20% width of the video.
    """
    try:
        # Step 1: Get video resolution
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[LOGO ERROR] Cannot open video.")
            return

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # Step 2: Resize logo to 20% of video width
        target_logo_width = int(video_width * 0.20)
        resized_logo_path = tempfile.mktemp(suffix=".png")

        subprocess.run([
            "ffmpeg", "-i", logo_path,
            "-vf", f"scale={target_logo_width}:-1",
            resized_logo_path, "-y"
        ], check=True)

        # Step 3: Overlay inside the video content area (top-right with 20px padding)
        output_path = video_path.replace("result.mp4", "result_with_logo.mp4")

        overlay_filter = "overlay=W-w-20:100"  # Top-right inside video, 20px padding

        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-i", resized_logo_path,
            "-filter_complex", overlay_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-c:a", "copy",
            output_path,
            "-y"
        ], check=True)

        # Replace the original result
        os.replace(output_path, video_path)
        print("[LOGO] Logo placed INSIDE video frame top-right. Job done.")

    except Exception as e:
        print(f"[LOGO ERROR] Overlay inside video failed: {e}")


# Simplified background processing without status field
async def process_face_swap_simple(
    uid: str,
    group_ids: list,
    enhancement_settings,
    loaded_dict: dict,
    db_session_maker
):
    """Background task to process face swap"""
    db = None
    try:
        processing_mode = "CPU" if not GPU_AVAILABLE else "GPU"
        print(f"[BACKGROUND] Starting face swap processing for {uid} using {processing_mode}")
        
        db = db_session_maker()
        
        # Load face data
        embeddings_file = os.path.join(UPLOAD_FOLDER, uid, 'face_embeddings.npy')
        bboxes_file = os.path.join(UPLOAD_FOLDER, uid, 'face_bboxes.npy')
        kps_file = os.path.join(UPLOAD_FOLDER, uid, 'face_kps.npy')

        all_embeddings = np.load(embeddings_file)
        all_bboxes = np.load(bboxes_file)
        all_kps = np.load(kps_file)
        all_face_info = loaded_dict['all_face_info']

        result_file_path = os.path.join(UPLOAD_FOLDER, uid, 'result.mp4')
        input_file_path = os.path.join(UPLOAD_FOLDER, uid, 'input.mp4')

        # Run face swap
        await asyncio.get_event_loop().run_in_executor(
            faceswap_executor,
            run_face_swap,
            uid,
            all_face_info,
            group_ids,
            all_embeddings,
            all_bboxes,
            all_kps,
            input_file_path,
            result_file_path
        )

        overlay_logo_on_video(result_file_path, LOGO_PATH)


        # Upload to S3
        s3_key = f"VFS/results/{uid}/result.mp4"
        s3_url = await asyncio.get_event_loop().run_in_executor(
            faceswap_executor,
            upload_file_to_s3,
            result_file_path,
            s3_key,
            "video/mp4"
        )

        # Update database
        record = db.query(VideoFaceSwap).filter(VideoFaceSwap.generation_id == uid).first()
        if record:
            record.swap_url = s3_url
            record.finished_at = datetime.now(timezone.utc)
            db.commit()

        # Cleanup
        cleanup_uploaded_files(uid)
        
        print(f"[BACKGROUND] Face swap completed successfully for {uid} using {processing_mode}")

    except Exception as e:
        print(f"[BACKGROUND ERROR] Face swap failed for {uid}: {str(e)}")
    finally:
        if db:
            db.close()

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

def get_ai_tool_id(db: Session, tool_name: str) -> str:
    """Retrieve the AI tool ID from the database based on tool name."""
    try:
        result = db.execute(
            text("""SELECT id FROM erosuniverse_interface.cms_app_aitools
                    WHERE name = :tool_name AND approved = true"""),
            {"tool_name": tool_name}
        ).fetchone()
        if result:
            return str(result[0])  # UUID to string
        else:
            raise Exception(f"AI tool '{tool_name}' not found or not approved")
    except Exception as e:
        print(f"[ERROR] Failed to get AI tool ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI tool ID: {str(e)}")






@router.get("/faceswap/status/{uid}")
async def check_face_swap_status(uid: str, db: Session = Depends(get_db)):
    """Get the status and result URL of the face swap process, including progress, dimensions, orientation, and duration."""
    try:
        def get_ist_now():
            return datetime.utcnow() + timedelta(hours=5, minutes=30)
 
        # Validate the UID format (check if it's a valid UUID)
        try:
            UUID(uid)  # This will raise a ValueError if the uid is not a valid UUID
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "error": "Invalid generation ID format", "message": "Invalid generation_id format. Must be a valid UUID."}
            )
 
        record = db.query(VideoFaceSwap).filter(VideoFaceSwap.generation_id == uid).first()
        if not record:
            return JSONResponse(
                status_code=404,
                content={"code": 404, "error": "Generation not found", "message": "Face swap generation not found"}
            )
 
        # --- Progress Reading ---
        progress = 0.0
        is_failed = False
        progress_path = os.path.join(UPLOAD_FOLDER, uid, 'progress.json')
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                    progress = progress_data.get('progress', 0.0)
                    # Check if there's an error status in progress file
                    is_failed = progress_data.get('status') == 'failed' or progress_data.get('error', False)
            except Exception:
                pass  # Ignore broken progress file

        # --- Check for failure conditions ---
        # Option 1: Check if record has a status field indicating failure
        if hasattr(record, 'status') and record.status == 'failed':
            is_failed = True
        
        # Option 2: Check if process has been running too long without completion
        if record.created_at:
            time_elapsed = get_ist_now() - record.created_at
            # Consider failed if running for more than 30 minutes without result
            if time_elapsed > timedelta(minutes=30) and not getattr(record, "swap_url", ""):
                is_failed = True
        
        # Option 3: Check for error file
        error_file_path = os.path.join(UPLOAD_FOLDER, uid, 'error.json')
        if os.path.exists(error_file_path):
            is_failed = True

        # --- Return failure response if process has failed ---
        if is_failed:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": {
                        "code": "FACE_SWAP_FAILED",
                        "message": "Unable to complete face swap, please try again after some time."
                    }
                }
            )
 
        # --- Swap URL and Metadata ---
        swap_url = getattr(record, "swap_url", "")
        has_result = bool(swap_url and swap_url.strip())
        s3_presigned_url = None
        file_url = None  # Add the unsigned URL
        result_dimensions = None
        orientation = None
 
        if has_result:
            s3_key = swap_url.replace(f"{S3_ENDPOINT}/{BUCKET_NAME}/", "")
            s3_presigned_url = generate_signed_url_for_video(s3_key)
            file_url = swap_url  # This is the unsigned S3 URL that gets stored in DB
 
            try:
                s3_client = get_s3_client()
                response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                video_bytes = response['Body'].read()
 
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
                    temp_vid.write(video_bytes)
                    temp_vid_path = temp_vid.name
 
                cap = cv2.VideoCapture(temp_vid_path)
                duration_seconds = None
 
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    orientation = "landscape" if width > height else "portrait"
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        duration_seconds = int(frame_count / fps)
 
                    result_dimensions = {
                        "width": width,
                        "height": height,
                        "orientation": orientation,
                        "duration": duration_seconds
                    }
                cap.release()
                os.unlink(temp_vid_path)
 
            except Exception as vid_meta_err:
                print(f"[WARNING] Could not extract video metadata: {vid_meta_err}")
 
            # --- Insert into erosapp_content if not exists ---
            try:
                content_exists = db.execute(text("""
                    SELECT 1 FROM erosuniverse_interface.erosapp_content WHERE content_id = :cid
                """), {"cid": uid}).fetchone()
 
                if not content_exists:
                    dimensions_str = f"{result_dimensions['width']}x{result_dimensions['height']}" if result_dimensions else ""
                    duration_val = result_dimensions.get("duration") if result_dimensions else None
 
                    db.execute(text("""
                        INSERT INTO erosuniverse_interface.erosapp_content (
                            content_id, content_type, file_format, file_url, status, visibility,
                            verification_status, user_id_id, created_at, updated_at, views,
                            share_count, orientation, dimensions, duration, aitools
                        ) VALUES (
                            :content_id, :content_type, :file_format, :file_url, :status, :visibility,
                            :verification_status, :user_id_id, :created_at, :updated_at, :views,
                            :share_count, :orientation, :dimensions, :duration, :aitools
                        )
                    """), {
                        "content_id": uid,
                        "content_type": "video",
                        "file_format": "mp4",
                        "file_url": swap_url,
                        "status": True,
                        "visibility": "public",
                        "verification_status": "pending",
                        "user_id_id": record.user_id,
                        "created_at": get_ist_now(),
                        "updated_at": get_ist_now(),
                        "views": 0,
                        "share_count": 0,
                        "orientation": orientation,
                        "dimensions": dimensions_str,
                        "duration": duration_val,
                        "aitools": "video face swap"
                    })
                    db.commit()
                    print(f"[INFO] Inserted content for UID: {uid}")
                else:
                    print(f"[INFO] Content for UID {uid} already exists.")
            except Exception as insert_err:
                print(f"[ERROR] DB insert failed for UID {uid}: {insert_err}")
 
        if progress >= 100.0 and not s3_presigned_url:
            progress = 99.0
 
        ai_tool_id = get_ai_tool_id(db, "Face Swap")
 
        response_data = {
            "generation_id": uid,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "progress": 100.0 if s3_presigned_url else progress,
            "status": "completed" if s3_presigned_url else "processing",
            "message": "Face swap completed successfully" if s3_presigned_url else "Face swap is being processed",
            "result_url": s3_presigned_url,
            "file_url": file_url,  # Add the unsigned S3 URL
            "result_video_dimensions": result_dimensions,
            "result_duration": f"{result_dimensions.get('duration')} sec" if result_dimensions and result_dimensions.get("duration") is not None else None,
            "content_id" : uid,
            "ai_tool_id": ai_tool_id
        }
 
        return JSONResponse(content=response_data)
 
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in check_face_swap_status: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"code": 400, "error": "Face swap failed", "message": "Please try again after some time"}
        )




 
@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str, db: Session = Depends(get_db)):
    """Get template information including presigned URL, credits, video dimensions, and swap type."""
    try:
        # Hardcoded values for credits, video_dimensions, and swap_type
        credits = 40
        video_dimensions = {
            "width": 1920,
            "height": 1080,
            "channels": 3
        }
        swap_type = "video-face-swap"
 
        # Generate the presigned URL for the template video
        presigned_template_url = generate_signed_url_for_video(f"VFS/Templates/{template_id}.mp4")
 
        # Return the response with the required data
        return JSONResponse(
            content={
                "template_id": template_id,
                "presigned_template_url": presigned_template_url,
                "credits": credits,
                "video_dimensions": video_dimensions,
                "swap_type": swap_type
            },
            status_code=200
        )
 
    except Exception as e:
        print(f"[ERROR] Failed to retrieve template info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve template information")
 
 
 
def generate_signed_url_for_video(s3_key: str, expiration: int = 3600):
    """Generate signed URL for video with inline content disposition"""
    try:
        s3_client = get_s3_client()
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': s3_key,
                'ResponseContentDisposition': 'inline',  # Ensures the file is displayed inline
                'ResponseContentType': 'video/mp4'  # Explicitly set content type for video files
            },
            ExpiresIn=expiration
        )
        return signed_url
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="S3 credentials are incorrect.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
       
        # Test S3 connection
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=BUCKET_NAME)
       
        return JSONResponse(
            content={
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database": "connected",
                "s3": "connected",
                "thread_pool": "active"
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            },
            status_code=503
        )
 
@router.options("/{path:path}")
async def options_route(path: str):
    """Handle OPTIONS requests for CORS"""
    return JSONResponse(
        content={"message": "OK"},
        headers=get_cors_headers()
    )
 
# =============================================================================
# OPTIONS HANDLER FOR CORS
# =============================================================================
 
# =============================================================================
# INCLUDE ROUTER AND ROOT ENDPOINT
# =============================================================================
 
# Include router in the app
app.include_router(router)
 
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        content={
            "message": "Production Video Face Swap API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "upload_video": "/uploadvideo/",
                "upload_faces": "/uploadnewfaces/{uid}/{group_id}",
                "face_swap": "/faceswap/{uid}",
                "health_check": "/health"
            }
        },
        status_code=200
    )
 
# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================


@app.on_event("startup")
async def startup_event():
    initialize_app()
    
    """Startup event handler"""
    print("[INFO] Starting Production Video Face Swap API...")
    print(f"[INFO] Thread Pool Size: {executor._max_workers}")
    print("[INFO] API startup completed successfully")
 
@app.on_event("shutdown")
async def shutdown_event():
    stop_cleanup_service()
    """Shutdown event handler"""
    
    print("[INFO] Shutting down Production Video Face Swap API...")
    executor.shutdown(wait=True)
    print("[INFO] API shutdown completed successfully")
 
# =============================================================================
# ERROR HANDLERS
# =============================================================================
 
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers=get_cors_headers()
    )
 
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    print(f"[ERROR] Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers=get_cors_headers()
    )
 

