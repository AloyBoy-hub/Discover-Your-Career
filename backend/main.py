from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from backend.resume_parser import parse_resume

app = FastAPI(title="Career Compass API")

# Allow CORS for potential frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Career Compass API is running"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save file temporarily
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Parse PDF
    try:
        data = parse_resume(file_path)
    except Exception as e:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")
    
    # Clean up file after parsing (or keep it if we need to store it)
    # For now, let's keep it but ideally we upload to S3.
    
    if not data:
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
        
    return {
        "status": "success",
        "file_id": file_id,
        "filename": file.filename,
        "data": data
    }

from pydantic import BaseModel
from typing import List
from backend.roadmap import generate_roadmap

class RoadmapRequest(BaseModel):
    skills: List[str]
    role: str

@app.post("/generate-roadmap")
async def create_roadmap(request: RoadmapRequest):
    return generate_roadmap(request.skills, request.role)
