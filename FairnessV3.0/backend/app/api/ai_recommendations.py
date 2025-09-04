from fastapi import APIRouter, HTTPException
from app.models.schemas import AIRecommendationRequest, AIRecommendation
from app.services.ai_recommendation_service import ai_recommendation_service

router = APIRouter()


@router.post("/generate", response_model=AIRecommendation)
async def generate_ai_recommendations(request: AIRecommendationRequest):
    """
    Generate AI-powered recommendations based on bias analysis
    
    This endpoint uses LLM (GPT-4 or Claude) to provide:
    - Specific bias reduction strategies
    - Model retraining approaches
    - Data collection improvements  
    - Monitoring and governance practices
    - Regulatory compliance considerations
    """
    try:
        recommendation = await ai_recommendation_service.generate_recommendations(request)
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.get("/recommendation/{job_id}", response_model=AIRecommendation)
async def get_ai_recommendation(job_id: str):
    """Get AI recommendation by job ID"""
    try:
        return ai_recommendation_service.get_recommendation(job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="AI recommendation not found")


@router.post("/chat")
async def chat_followup(
    recommendation_job_id: str,
    question: str
):
    """
    Ask follow-up questions about the AI recommendations
    
    This allows users to get clarification or additional details
    about the generated recommendations.
    """
    try:
        # Get the original recommendation
        original_recommendation = ai_recommendation_service.get_recommendation(recommendation_job_id)
        
        # Create a follow-up context
        followup_prompt = f"""
Based on the previous fairness assessment recommendations, please answer this follow-up question:

Question: {question}

Previous recommendations context:
{original_recommendation.recommendations[:1000]}...

Please provide a concise, helpful answer focused on the user's specific question.
"""
        
        # Generate follow-up response (mock for now)
        if ai_recommendation_service.use_mock_responses:
            followup_response = f"""
Based on your question about "{question}", here's my recommendation:

This relates to the fairness assessment we discussed. The key points to consider are:

• Review the specific bias metrics that were identified in your analysis
• Consider the context of your model's application domain
• Implement changes incrementally and measure their impact
• Ensure any changes align with your business requirements and ethical guidelines

For more specific guidance, please refer to the detailed recommendations provided in your assessment report.
"""
        else:
            # In production, this would call the AI API with the follow-up prompt
            followup_response = "AI follow-up feature requires API configuration"
        
        return {
            "question": question,
            "answer": followup_response,
            "recommendation_job_id": recommendation_job_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process follow-up: {str(e)}")


@router.get("/status")
async def get_ai_service_status():
    """Get AI service configuration status"""
    return {
        "ai_service_available": True,
        "using_mock_responses": ai_recommendation_service.use_mock_responses,
        "openai_configured": bool(ai_recommendation_service.openai_api_key),
        "anthropic_configured": bool(ai_recommendation_service.anthropic_api_key),
        "message": "Mock responses enabled - configure OPENAI_API_KEY or ANTHROPIC_API_KEY for full AI features"
        if ai_recommendation_service.use_mock_responses else "AI service fully configured"
    }


@router.get("/list")
async def list_ai_recommendations():
    """List all AI recommendations"""
    recommendations = []
    for job_id, recommendation in ai_recommendation_service.ai_recommendations.items():
        recommendations.append({
            "job_id": job_id,
            "generated_at": recommendation.generated_at,
            "confidence_score": recommendation.confidence_score,
            "strategies_count": len(recommendation.bias_reduction_strategies),
            "approaches_count": len(recommendation.model_retraining_approaches)
        })
    return {"recommendations": recommendations}
