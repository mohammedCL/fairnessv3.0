from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional

from app.models.schemas import MitigationJob, MitigationResults, MitigationStrategy
from app.services.mitigation_service import mitigation_service

router = APIRouter()


@router.post("/start", response_model=MitigationJob)
async def start_mitigation(
    analysis_job_id: str,
    strategy: MitigationStrategy,
    strategy_params: Optional[Dict[str, Any]] = None
):
    """
    Start bias mitigation job
    
    Available strategies:
    - preprocessing: Data resampling, feature selection, data augmentation
    - inprocessing: Fairness constraints, adversarial debiasing, regularization  
    - postprocessing: Threshold optimization, calibration adjustment
    """
    try:
        job = await mitigation_service.start_mitigation(
            analysis_job_id=analysis_job_id,
            strategy=strategy,
            strategy_params=strategy_params or {}
        )
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start mitigation: {str(e)}")


@router.get("/job/{job_id}", response_model=MitigationJob)
async def get_mitigation_job(job_id: str):
    """Get mitigation job status and progress"""
    return mitigation_service.get_mitigation_job(job_id)


@router.get("/results/{job_id}", response_model=MitigationResults)
async def get_mitigation_results(job_id: str):
    """Get complete mitigation results"""
    return mitigation_service.get_mitigation_results(job_id)


@router.get("/comparison/{job_id}")
async def get_before_after_comparison(job_id: str):
    """Get before/after bias metrics comparison"""
    results = mitigation_service.get_mitigation_results(job_id)
    return {
        "job_id": job_id,
        "strategy_applied": results.strategy_applied,
        "before_metrics": results.before_metrics,
        "after_metrics": results.after_metrics,
        "improvement_summary": results.improvement_summary,
        "fairness_improvement": results.fairness_improvement
    }


@router.get("/improvement-summary/{job_id}")
async def get_improvement_summary(job_id: str):
    """Get mitigation improvement summary"""
    results = mitigation_service.get_mitigation_results(job_id)
    return {
        "job_id": job_id,
        "overall_improvement": results.fairness_improvement,
        "strategy_used": results.strategy_applied,
        "metric_improvements": results.improvement_summary,
        "mitigation_successful": results.fairness_improvement > 0
    }


@router.get("/strategies")
async def get_available_strategies():
    """Get list of available mitigation strategies"""
    return {
        "strategies": [
            {
                "name": "preprocessing",
                "display_name": "Preprocessing",
                "description": "Apply bias mitigation during data preprocessing",
                "techniques": [
                    "Data reweighing",
                    "Disparate impact remover", 
                    "Data augmentation for underrepresented groups"
                ],
                "parameters": {
                    "use_reweighing": {"type": "boolean", "default": True},
                    "use_disparate_impact_remover": {"type": "boolean", "default": True},
                    "use_data_augmentation": {"type": "boolean", "default": True}
                }
            },
            {
                "name": "inprocessing",
                "display_name": "In-processing",
                "description": "Apply fairness constraints during model training",
                "techniques": [
                    "Fairness constraints",
                    "Regularization for fairness",
                    "Adversarial debiasing"
                ],
                "parameters": {
                    "fairness_penalty": {"type": "float", "default": 0.1, "range": [0.01, 1.0]}
                }
            },
            {
                "name": "postprocessing", 
                "display_name": "Post-processing",
                "description": "Adjust model predictions to achieve fairness",
                "techniques": [
                    "Threshold optimization",
                    "Calibration adjustment",
                    "Equalized odds post-processing"
                ],
                "parameters": {
                    "equalized_odds": {"type": "boolean", "default": True},
                    "threshold_optimization": {"type": "boolean", "default": True}
                }
            }
        ]
    }


@router.get("/list")
async def list_mitigation_jobs():
    """List all mitigation jobs"""
    jobs = []
    for job_id, job in mitigation_service.mitigation_jobs.items():
        jobs.append({
            "job_id": job_id,
            "analysis_job_id": job.analysis_job_id,
            "strategy": job.strategy,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        })
    return {"jobs": jobs}
