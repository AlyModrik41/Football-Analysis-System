from fastapi import FastAPI, UploadFile, File
import shutil
import os
from app.pipeline import run_analysis
from app.models import MatchStats

app= FastAPI(title='Football Analytics System')

UPLOAD_DIR='uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/analyze', response_model=MatchStats)
async def analyze_video(file:UploadFile=File(...)):
    file_path=os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results= run_analysis(file_path)

    return results
    
