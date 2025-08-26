from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, MetaData, text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from databases import Database
from datetime import datetime
import os
import sys
 
import torch

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  
 
# Add your router
sys.path.append(os.path.dirname(__file__))                                
from face_swap.api import router
 
# === Database Configuration ===
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:enePuf77vyIHTZGk@103.114.153.167:49502/postgres?sslmode=require"
)
 
# === SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
 
# === Schema setup
SCHEMA_NAME = "creator_tools"
metadata = MetaData(schema=SCHEMA_NAME)
Base = declarative_base(metadata=metadata)  
 
# === Async support (optional)
database = Database(DATABASE_URL)
 
# === Ensure schema exists
with engine.connect() as conn:
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
    conn.commit()
 
# === DB model
class VideoFaceSwap(Base):
    __tablename__ = "videofaceswap"
    __table_args__ = {"schema": SCHEMA_NAME}
 
    generation_id = Column(String, primary_key=True, index=True)
    # user_id = Column(Integer, nullable=False)
    user_id = Column(String(255), nullable=False)
    template_id = Column(Integer, nullable=False)
    detected_face_url = Column(String, nullable=True)
    target_url = Column(String, nullable=True)
    swap_url = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    finished_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    credits = Column(Integer, nullable=False, default=20)

    
 
 
# === DB model for fetching user IDs ===
from sqlalchemy import select
 
# Function to get allowed user IDs from the database
def get_allowed_user_ids(db: Session):
    # Query to fetch user IDs from the 'erosapp_user' table in the 'erosuniverse_interface' schema
    result = db.execute(select([VideoFaceSwap.user_id]).distinct().select_from(
        f"{SCHEMA_NAME}.erosapp_user").filter(f"{SCHEMA_NAME}.erosapp_user.is_active == True"))
    allowed_user_ids = {row[0] for row in result.fetchall()}
    return allowed_user_ids
 
  
 
# === Create tables
Base.metadata.create_all(bind=engine)
 
# === FastAPI app
app = FastAPI()
 
# === CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# === Dependency for DB session with IST
def get_db():
    db = SessionLocal()
    try:
        db.execute(text("SET TIME ZONE 'Asia/Kolkata';"))  
        yield db
    finally:
        db.close()
 
# === Include router
app.include_router(router)
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=1200                                                                                        ,
    )