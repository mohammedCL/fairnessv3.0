"""
API endpoints for comprehensive bias mitigation
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.schemas import MitigationJob
from app.services.comprehensive_mitigation_service import comprehensive_mitigation_service

router = APIRouter()


@router.post("/comprehensive/start", response_model=MitigationJob)
async def start_comprehensive_mitigation(analysis_job_id: str):
    """
    Start comprehensive bias mitigation that applies all available strategies
    
    This endpoint will:
    1. Calculate baseline bias metrics before mitigation
    2. Apply all available bias mitigation strategies sequentially:
       - Preprocessing: Reweighing, Disparate Impact Remover, Data Augmentation
       - In-processing: Fairness Regularization, Adversarial Debiasing
       - Post-processing: Threshold Optimization, Calibration Adjustment, Equalized Odds
    3. Evaluate the effectiveness of each strategy
    4. Compare bias metrics before and after each mitigation
    5. Identify the best performing strategy
    6. Provide comprehensive results for frontend display
    """
    try:
        job = await comprehensive_mitigation_service.start_comprehensive_mitigation(
            analysis_job_id=analysis_job_id
        )
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start comprehensive mitigation: {str(e)}")


@router.get("/comprehensive/job/{job_id}")
async def get_comprehensive_mitigation_job(job_id: str):
    """Get comprehensive mitigation job status and progress"""
    try:
        job = comprehensive_mitigation_service.get_mitigation_job(job_id)
        return {
            "job_id": job.job_id,
            "analysis_job_id": job.analysis_job_id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/results/{job_id}")
async def get_comprehensive_mitigation_results(job_id: str):
    """
    Get comprehensive mitigation results in the format expected by frontend
    
    Returns:
    {
        "bias_before": {"metric_name": value, ...},
        "bias_after": [
            {
                "strategy": "Strategy Name",
                "metrics": {"metric_name": value, ...},
                "fairness_score": value,
                "model_performance": {"accuracy": value, ...},
                "execution_time": value
            },
            ...
        ],
        "best_strategy": "Strategy Name",
        "improvements": {"metric_name": percentage_improvement, ...},
        "overall_fairness_improvement": value,
        "execution_summary": {...},
        "recommendations": [...]
    }
    """
    try:
        formatted_results = comprehensive_mitigation_service.format_results_for_frontend(job_id)
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/comparison/{job_id}")
async def get_bias_comparison(job_id: str):
    """
    Get before vs after bias comparison for all strategies
    
    Returns data optimized for comparison tables and visualizations
    """
    try:
        results = comprehensive_mitigation_service.get_comprehensive_results(job_id)
        
        # Create comparison data structure
        comparison_data = {
            "baseline_metrics": results.bias_before,
            "strategy_comparisons": [],
            "best_strategy": results.best_strategy,
            "summary": {
                "total_strategies_tested": len(results.bias_after),
                "successful_strategies": len([r for r in results.bias_after if r.success]),
                "overall_improvement": results.overall_fairness_improvement
            }
        }
        
        # Add detailed comparison for each strategy
        for result in results.bias_after:
            if result.success:
                strategy_comparison = {
                    "strategy_name": result.strategy_name,
                    "strategy_type": result.strategy_type,
                    "fairness_score": result.fairness_score,
                    "model_performance": result.model_performance,
                    "metric_changes": {},
                    "improvement_percentage": result.improvement_percentage
                }
                
                # Calculate metric changes
                for metric_name, after_value in result.metrics.items():
                    if metric_name in results.bias_before:
                        before_value = results.bias_before[metric_name]
                        change = after_value - before_value
                        strategy_comparison["metric_changes"][metric_name] = {
                            "before": before_value,
                            "after": after_value,
                            "change": change,
                            "improvement_percentage": result.improvement_percentage.get(metric_name, 0)
                        }
                
                comparison_data["strategy_comparisons"].append(strategy_comparison)
        
        return comparison_data
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/best-strategy/{job_id}")
async def get_best_strategy_details(job_id: str):
    """Get detailed information about the best performing strategy"""
    try:
        results = comprehensive_mitigation_service.get_comprehensive_results(job_id)
        
        # Find the best strategy result
        best_result = None
        for result in results.bias_after:
            if result.success and result.strategy_name == results.best_strategy:
                best_result = result
                break
        
        if not best_result:
            return {"error": "Best strategy not found or failed"}
        
        return {
            "strategy_name": best_result.strategy_name,
            "strategy_type": best_result.strategy_type,
            "fairness_score": best_result.fairness_score,
            "model_performance": best_result.model_performance,
            "execution_time": best_result.execution_time,
            "metrics": best_result.metrics,
            "improvement_percentage": best_result.improvement_percentage,
            "recommendations": [
                rec for rec in results.recommendations 
                if best_result.strategy_name in rec
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/fairness-scores/{job_id}")
async def get_fairness_scores_comparison(job_id: str):
    """Get fairness scores for all strategies for easy comparison"""
    try:
        results = comprehensive_mitigation_service.get_comprehensive_results(job_id)
        
        fairness_scores = []
        for result in results.bias_after:
            if result.success:
                fairness_scores.append({
                    "strategy": result.strategy_name,
                    "strategy_type": result.strategy_type,
                    "fairness_score": result.fairness_score,
                    "is_best": result.strategy_name == results.best_strategy
                })
        
        # Sort by fairness score descending
        fairness_scores.sort(key=lambda x: x["fairness_score"], reverse=True)
        
        return {
            "fairness_scores": fairness_scores,
            "best_score": fairness_scores[0]["fairness_score"] if fairness_scores else 0,
            "improvement_from_baseline": results.overall_fairness_improvement
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/performance-tradeoffs/{job_id}")
async def get_performance_tradeoffs(job_id: str):
    """
    Analyze trade-offs between fairness and model performance
    """
    try:
        results = comprehensive_mitigation_service.get_comprehensive_results(job_id)
        
        tradeoff_analysis = []
        for result in results.bias_after:
            if result.success:
                tradeoff_analysis.append({
                    "strategy": result.strategy_name,
                    "strategy_type": result.strategy_type,
                    "fairness_score": result.fairness_score,
                    "accuracy": result.model_performance.get("accuracy", 0),
                    "precision": result.model_performance.get("precision", 0),
                    "recall": result.model_performance.get("recall", 0),
                    "f1_score": result.model_performance.get("f1_score", 0),
                    "fairness_performance_ratio": (
                        result.fairness_score / (result.model_performance.get("accuracy", 1) * 100)
                        if result.model_performance.get("accuracy", 0) > 0 else 0
                    )
                })
        
        # Find best balanced strategy (highest fairness-performance ratio)
        best_balanced = max(tradeoff_analysis, key=lambda x: x["fairness_performance_ratio"]) if tradeoff_analysis else None
        
        return {
            "tradeoff_analysis": tradeoff_analysis,
            "best_balanced_strategy": best_balanced["strategy"] if best_balanced else None,
            "recommendations": {
                "for_fairness": results.best_strategy,
                "for_performance": max(tradeoff_analysis, key=lambda x: x["accuracy"])["strategy"] if tradeoff_analysis else None,
                "for_balance": best_balanced["strategy"] if best_balanced else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/execution-summary/{job_id}")
async def get_execution_summary(job_id: str):
    """Get detailed execution summary and recommendations"""
    try:
        results = comprehensive_mitigation_service.get_comprehensive_results(job_id)
        
        return {
            "job_id": job_id,
            "execution_summary": results.execution_summary,
            "recommendations": results.recommendations,
            "best_strategy": results.best_strategy,
            "overall_improvement": results.overall_fairness_improvement,
            "strategy_success_rate": (
                results.execution_summary["successful_strategies"] / 
                results.execution_summary["total_strategies"] * 100
                if results.execution_summary["total_strategies"] > 0 else 0
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comprehensive/available-strategies")
async def get_comprehensive_strategies():
    """Get information about all available comprehensive mitigation strategies"""
    return {
        "strategies": [
            {
                "key": "reweighing",
                "name": "Data Reweighing",
                "type": "preprocessing",
                "description": "Reweight training samples to balance sensitive groups",
                "ideal_for": "Datasets with imbalanced sensitive groups"
            },
            {
                "key": "disparate_impact_remover",
                "name": "Disparate Impact Remover",
                "type": "preprocessing",
                "description": "Remove features that cause disparate impact",
                "ideal_for": "When direct discrimination through features is suspected"
            },
            {
                "key": "data_augmentation",
                "name": "Data Augmentation",
                "type": "preprocessing",
                "description": "Augment underrepresented groups with synthetic data",
                "ideal_for": "Small datasets with underrepresented groups"
            },
            {
                "key": "fairness_regularization",
                "name": "Fairness Regularization",
                "type": "inprocessing",
                "description": "Add fairness penalty to model training",
                "ideal_for": "When you can retrain the model with fairness constraints"
            },
            {
                "key": "adversarial_debiasing",
                "name": "Adversarial Debiasing",
                "type": "inprocessing",
                "description": "Use adversarial training to remove bias",
                "ideal_for": "Complex models where traditional methods fail"
            },
            {
                "key": "threshold_optimization",
                "name": "Threshold Optimization",
                "type": "postprocessing",
                "description": "Optimize decision thresholds for each group",
                "ideal_for": "When model retraining is not possible"
            },
            {
                "key": "calibration_adjustment",
                "name": "Calibration Adjustment",
                "type": "postprocessing",
                "description": "Calibrate predictions to ensure fairness",
                "ideal_for": "Probability-based models with calibration issues"
            },
            {
                "key": "equalized_odds_postprocessing",
                "name": "Equalized Odds Post-processing",
                "type": "postprocessing",
                "description": "Adjust predictions to achieve equalized odds",
                "ideal_for": "When equalized odds is the primary fairness goal"
            }
        ],
        "strategy_types": {
            "preprocessing": {
                "name": "Preprocessing",
                "description": "Apply bias mitigation during data preprocessing",
                "pros": ["Addresses bias at the source", "Works with any model"],
                "cons": ["May lose important information", "Requires data access"]
            },
            "inprocessing": {
                "name": "In-processing",
                "description": "Apply fairness constraints during model training",
                "pros": ["Integrates fairness into model", "Can optimize for fairness and accuracy"],
                "cons": ["Requires model retraining", "Model-specific implementations"]
            },
            "postprocessing": {
                "name": "Post-processing",
                "description": "Adjust model predictions to achieve fairness",
                "pros": ["Works with existing models", "Quick to implement"],
                "cons": ["May hurt model performance", "Doesn't address root causes"]
            }
        }
    }
