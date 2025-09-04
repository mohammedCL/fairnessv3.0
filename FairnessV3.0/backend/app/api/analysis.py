from fastapi import APIRouter, HTTPException
from typing import Optional

from app.models.schemas import AnalysisJob, AnalysisResults
from app.services.analysis_service import analysis_service

router = APIRouter()


@router.post("/start", response_model=AnalysisJob)
async def start_analysis(
    model_upload_id: str,
    train_dataset_upload_id: str,
    test_dataset_upload_id: Optional[str] = None
):
    """
    Start bias analysis job
    
    This will:
    1. Analyze the uploaded model
    2. Detect sensitive features in the dataset
    3. Perform bias detection (direct, proxy, and model bias)
    4. Calculate fairness scores
    5. Generate visualizations
    """
    try:
        job = await analysis_service.start_analysis(
            model_upload_id=model_upload_id,
            train_dataset_upload_id=train_dataset_upload_id,
            test_dataset_upload_id=test_dataset_upload_id
        )
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@router.get("/job/{job_id}", response_model=AnalysisJob)
async def get_analysis_job(job_id: str):
    """Get analysis job status and progress"""
    return analysis_service.get_analysis_job(job_id)


@router.get("/results/{job_id}", response_model=AnalysisResults)
async def get_analysis_results(job_id: str):
    """Get complete analysis results"""
    return analysis_service.get_analysis_results(job_id)


@router.get("/bias-metrics/{job_id}")
async def get_bias_metrics(job_id: str):
    """Get bias metrics for a completed analysis"""
    results = analysis_service.get_analysis_results(job_id)
    return {
        "job_id": job_id,
        "bias_metrics": results.bias_metrics,
        "sensitive_features": results.sensitive_features
    }


@router.get("/fairness-score/{job_id}")
async def get_fairness_score(job_id: str):
    """Get fairness score for a completed analysis"""
    results = analysis_service.get_analysis_results(job_id)
    return {
        "job_id": job_id,
        "fairness_score": results.fairness_score,
        "model_info": results.model_info
    }


@router.get("/visualizations/{job_id}")
async def get_visualizations(job_id: str):
    """Get visualization charts for a completed analysis"""
    results = analysis_service.get_analysis_results(job_id)
    return {
        "job_id": job_id,
        "visualizations": results.visualizations
    }


@router.get("/summary/{job_id}")
async def get_analysis_summary(job_id: str):
    """Get analysis summary report"""
    results = analysis_service.get_analysis_results(job_id)
    return {
        "job_id": job_id,
        "summary": results.analysis_summary,
        "fairness_level": results.fairness_score.fairness_level,
        "overall_score": results.fairness_score.overall_score
    }


@router.get("/list")
async def list_analysis_jobs():
    """List all analysis jobs"""
    jobs = []
    for job_id, job in analysis_service.analysis_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "model_upload_id": job.model_upload_id,
            "train_dataset_upload_id": job.train_dataset_upload_id
        })
    return {"jobs": jobs}
