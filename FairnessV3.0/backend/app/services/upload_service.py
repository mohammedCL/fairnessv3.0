import os
import uuid
import shutil
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from app.models.schemas import UploadResponse, FileType, ModelType, TaskType
from app.utils.file_validators import validate_model_file, validate_dataset_file, detect_model_info


class UploadService:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.models_dir = os.path.join(upload_dir, "models")
        self.datasets_dir = os.path.join(upload_dir, "datasets")
        self.temp_dir = os.path.join(upload_dir, "temp")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # In-memory storage for upload metadata (use database in production)
        self.uploads: Dict[str, Dict] = {}
    
    async def upload_model(self, file: UploadFile) -> UploadResponse:
        """Upload and validate a model file"""
        try:
            # Generate unique upload ID
            upload_id = str(uuid.uuid4())
            
            # Validate file extension
            if not file.filename.endswith(('.pkl', '.joblib', '.onnx')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid model file format. Supported: .pkl, .joblib, .onnx"
                )
            
            # Save file
            file_path = os.path.join(self.models_dir, f"{upload_id}_{file.filename}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Validate model file
            is_valid, validation_info = validate_model_file(file_path)
            if not is_valid:
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model file: {validation_info}"
                )
            
            # Store upload metadata
            self.uploads[upload_id] = {
                "filename": file.filename,
                "file_path": file_path,
                "file_type": FileType.MODEL,
                "file_size": os.path.getsize(file_path),
                "validation_info": validation_info,
                "status": "uploaded"
            }
            
            return UploadResponse(
                upload_id=upload_id,
                filename=file.filename,
                file_type=FileType.MODEL,
                file_size=os.path.getsize(file_path)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def upload_dataset(self, file: UploadFile, dataset_type: str) -> UploadResponse:
        """Upload and validate a dataset file"""
        try:
            # Generate unique upload ID
            upload_id = str(uuid.uuid4())
            
            # Validate file extension
            if not file.filename.endswith(('.csv', '.json', '.parquet', '.xlsx')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid dataset format. Supported: .csv, .json, .parquet, .xlsx"
                )
            
            # Save file
            file_path = os.path.join(self.datasets_dir, f"{upload_id}_{file.filename}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Validate dataset file
            is_valid, validation_info = validate_dataset_file(file_path)
            if not is_valid:
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid dataset file: {validation_info}"
                )
            
            # Determine file type
            file_type = FileType.TRAIN_DATASET if dataset_type == "train" else FileType.TEST_DATASET
            
            # Store upload metadata
            self.uploads[upload_id] = {
                "filename": file.filename,
                "file_path": file_path,
                "file_type": file_type,
                "file_size": os.path.getsize(file_path),
                "validation_info": validation_info,
                "status": "uploaded"
            }
            
            return UploadResponse(
                upload_id=upload_id,
                filename=file.filename,
                file_type=file_type,
                file_size=os.path.getsize(file_path)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    def get_upload_status(self, upload_id: str) -> Dict:
        """Get upload status and metadata"""
        if upload_id not in self.uploads:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return self.uploads[upload_id]
    
    def get_file_path(self, upload_id: str) -> str:
        """Get file path for uploaded file"""
        if upload_id not in self.uploads:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return self.uploads[upload_id]["file_path"]
    
    def load_model(self, upload_id: str):
        """Load model from uploaded file"""
        file_path = self.get_file_path(upload_id)
        
        print(f"Loading model from: {file_path}")
        
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                    print(f"Loaded model type: {type(model)}")
                    print(f"Model has predict: {hasattr(model, 'predict')}")
                    print(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")
                    return model
            elif file_path.endswith('.joblib'):
                model = joblib.load(file_path)
                print(f"Loaded model type: {type(model)}")
                print(f"Model has predict: {hasattr(model, 'predict')}")
                print(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")
                return model
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported model format for loading"
                )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def load_dataset(self, upload_id: str) -> pd.DataFrame:
        """Load dataset from uploaded file"""
        file_path = self.get_file_path(upload_id)
        
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            elif file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported dataset format"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load dataset: {str(e)}"
            )
    
    def cleanup_upload(self, upload_id: str):
        """Clean up uploaded file"""
        if upload_id in self.uploads:
            file_path = self.uploads[upload_id]["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
            del self.uploads[upload_id]


# Global upload service instance
upload_service = UploadService()
