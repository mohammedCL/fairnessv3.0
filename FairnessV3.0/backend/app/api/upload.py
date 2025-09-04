from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Dict

from app.models.schemas import UploadResponse, UploadStatus
from app.services.upload_service import upload_service

router = APIRouter()


@router.post("/model", response_model=UploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a trained model file
    
    Supported formats:
    - .pkl (pickle)
    - .joblib (joblib)
    - .onnx (ONNX)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    return await upload_service.upload_model(file)


@router.post("/train-dataset", response_model=UploadResponse)
async def upload_train_dataset(file: UploadFile = File(...)):
    """
    Upload training dataset
    
    Supported formats:
    - .csv
    - .json
    - .parquet
    - .xlsx
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    return await upload_service.upload_dataset(file, "train")


@router.post("/test-dataset", response_model=UploadResponse)
async def upload_test_dataset(file: UploadFile = File(...)):
    """
    Upload test dataset
    
    Supported formats:
    - .csv
    - .json
    - .parquet
    - .xlsx
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    return await upload_service.upload_dataset(file, "test")


@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get upload status and metadata"""
    return upload_service.get_upload_status(upload_id)


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str):
    """Delete uploaded file"""
    try:
        upload_service.cleanup_upload(upload_id)
        return {"message": "Upload deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete upload: {str(e)}")


@router.get("/list")
async def list_uploads():
    """List all uploads"""
    uploads = []
    for upload_id, metadata in upload_service.uploads.items():
        uploads.append({
            "upload_id": upload_id,
            "filename": metadata["filename"],
            "file_type": metadata["file_type"],
            "file_size": metadata["file_size"],
            "status": metadata["status"]
        })
    return {"uploads": uploads}
