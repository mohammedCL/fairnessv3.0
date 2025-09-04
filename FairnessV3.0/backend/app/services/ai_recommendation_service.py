import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

from app.models.schemas import (
    AIRecommendationRequest, AIRecommendation, AnalysisResults, 
    MitigationResults, ModelType, TaskType
)
from app.services.analysis_service import analysis_service
from app.services.mitigation_service import mitigation_service


class AIRecommendationService:
    def __init__(self):
        # In-memory storage for AI recommendations
        self.ai_recommendations: Dict[str, AIRecommendation] = {}
        
        # Configure AI providers (you'll need to set these environment variables)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Use mock responses if no API keys are configured
        self.use_mock_responses = not (self.openai_api_key or self.anthropic_api_key)
    
    async def generate_recommendations(self, request: AIRecommendationRequest) -> AIRecommendation:
        """Generate AI-powered recommendations"""
        try:
            # Get analysis results
            analysis_results = analysis_service.get_analysis_results(request.analysis_job_id)
            
            # Get mitigation results if available
            mitigation_results = None
            if request.mitigation_job_id:
                mitigation_results = mitigation_service.get_mitigation_results(request.mitigation_job_id)
            
            # Create context for AI
            context = self._create_ai_context(analysis_results, mitigation_results, request.additional_context)
            
            # Generate recommendations using AI
            if self.use_mock_responses:
                recommendations_text = self._generate_mock_recommendations(context)
            else:
                recommendations_text = await self._call_ai_api(context)
            
            # Parse and structure the response
            structured_recommendations = self._parse_ai_response(recommendations_text, context)
            
            # Store and return recommendation
            recommendation = AIRecommendation(
                recommendations=recommendations_text,
                bias_reduction_strategies=structured_recommendations["bias_reduction_strategies"],
                model_retraining_approaches=structured_recommendations["model_retraining_approaches"],
                data_collection_improvements=structured_recommendations["data_collection_improvements"],
                monitoring_practices=structured_recommendations["monitoring_practices"],
                compliance_considerations=structured_recommendations["compliance_considerations"],
                confidence_score=structured_recommendations["confidence_score"]
            )
            
            self.ai_recommendations[recommendation.job_id] = recommendation
            return recommendation
            
        except Exception as e:
            raise Exception(f"Failed to generate AI recommendations: {str(e)}")
    
    def _create_ai_context(self, analysis_results: AnalysisResults, 
                          mitigation_results: Optional[MitigationResults],
                          additional_context: Optional[str]) -> Dict[str, Any]:
        """Create context for AI recommendation generation"""
        
        context = {
            "model_type": analysis_results.model_info.model_type.value,
            "task_type": analysis_results.model_info.task_type.value if analysis_results.model_info.task_type else "unknown",
            "framework": analysis_results.model_info.framework,
            "n_features": analysis_results.model_info.n_features,
            "n_classes": analysis_results.model_info.n_classes,
            "dataset_size": "unknown",  # Could be extracted from upload metadata
            "bias_score": analysis_results.fairness_score.bias_score,
            "fairness_score": analysis_results.fairness_score.overall_score,
            "fairness_level": analysis_results.fairness_score.fairness_level,
            "sensitive_features": [sf.feature_name for sf in analysis_results.sensitive_features],
            "bias_metrics": [
                {
                    "name": bm.metric_name,
                    "value": bm.value,
                    "is_biased": bm.is_biased,
                    "severity": bm.severity
                }
                for bm in analysis_results.bias_metrics
            ],
            "high_bias_areas": [
                bm.metric_name for bm in analysis_results.bias_metrics 
                if bm.severity == "high"
            ]
        }
        
        if mitigation_results:
            context.update({
                "mitigation_strategy": mitigation_results.strategy_applied.value,
                "mitigation_applied": True,
                "post_mitigation_bias": np.mean([abs(m.value) for m in mitigation_results.after_metrics]) if mitigation_results.after_metrics else 0,
                "fairness_improvement": mitigation_results.fairness_improvement,
                "improvement_summary": mitigation_results.improvement_summary
            })
        else:
            context.update({
                "mitigation_applied": False,
                "post_mitigation_bias": context["bias_score"]
            })
        
        if additional_context:
            context["additional_context"] = additional_context
        
        return context
    
    async def _call_ai_api(self, context: Dict[str, Any]) -> str:
        """Call AI API (OpenAI or Anthropic) for recommendations"""
        
        prompt = self._create_ai_prompt(context)
        
        try:
            if self.openai_api_key:
                return await self._call_openai(prompt)
            elif self.anthropic_api_key:
                return await self._call_anthropic(prompt)
            else:
                return self._generate_mock_recommendations(context)
        except Exception as e:
            print(f"AI API call failed: {str(e)}")
            return self._generate_mock_recommendations(context)
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI fairness consultant providing detailed recommendations for bias mitigation in machine learning models."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            raise e
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Anthropic API error: {str(e)}")
            raise e
    
    def _create_ai_prompt(self, context: Dict[str, Any]) -> str:
        """Create detailed AI prompt for recommendations"""
        
        prompt = f"""
Analyze this ML fairness assessment and provide comprehensive recommendations:

## Model Information
- Model Type: {context['model_type']}
- Framework: {context['framework']}
- Task Type: {context['task_type']}
- Number of Features: {context['n_features']}
- Number of Classes: {context.get('n_classes', 'N/A')}

## Fairness Assessment Results
- Overall Fairness Score: {context['fairness_score']:.1f}/100
- Bias Score: {context['bias_score']:.1f}/100
- Fairness Level: {context['fairness_level']}
- Sensitive Features Detected: {', '.join(context['sensitive_features'])}

## Bias Analysis
High-severity bias areas: {', '.join(context['high_bias_areas']) if context['high_bias_areas'] else 'None'}

Detailed bias metrics:
{self._format_bias_metrics(context['bias_metrics'])}

## Mitigation Status
- Mitigation Applied: {context['mitigation_applied']}
"""

        if context['mitigation_applied']:
            prompt += f"""
- Strategy Used: {context['mitigation_strategy']}
- Post-mitigation Bias Score: {context['post_mitigation_bias']:.1f}
- Fairness Improvement: {context['fairness_improvement']:.1f}%
"""

        if context.get('additional_context'):
            prompt += f"""
## Additional Context
{context['additional_context']}
"""

        prompt += """

Please provide specific, actionable recommendations in the following categories:

## 1. Bias Reduction Strategies
Provide 3-5 specific strategies to further reduce bias, considering the current model type and detected bias patterns.

## 2. Model Retraining Approaches  
Suggest concrete approaches for retraining the model to improve fairness, including algorithm choices and hyperparameters.

## 3. Data Collection Improvements
Recommend specific improvements to data collection and dataset composition to address identified bias sources.

## 4. Monitoring and Governance Practices
Outline ongoing monitoring practices and governance frameworks to maintain fairness over time.

## 5. Regulatory Compliance Considerations
Highlight relevant regulations and compliance considerations based on the detected bias patterns and model application domain.

Format your response clearly with actionable bullet points under each category. Be specific and practical in your recommendations.
"""
        
        return prompt
    
    def _format_bias_metrics(self, bias_metrics: list) -> str:
        """Format bias metrics for AI prompt"""
        if not bias_metrics:
            return "No significant bias detected."
        
        formatted = []
        for metric in bias_metrics[:5]:  # Limit to top 5 metrics
            status = "⚠️ BIASED" if metric["is_biased"] else "✅ OK"
            formatted.append(f"- {metric['name']}: {metric['value']:.3f} ({metric['severity']} severity) {status}")
        
        return "\n".join(formatted)
    
    def _generate_mock_recommendations(self, context: Dict[str, Any]) -> str:
        """Generate mock recommendations when AI APIs are not available"""
        
        fairness_score = context['fairness_score']
        bias_score = context['bias_score']
        model_type = context['model_type']
        sensitive_features = context['sensitive_features']
        
        mock_response = f"""
# AI Fairness Assessment Recommendations

Based on your model analysis showing a fairness score of {fairness_score:.1f}/100 and bias score of {bias_score:.1f}/100, here are my recommendations:

## 1. Bias Reduction Strategies

"""
        
        if bias_score > 50:
            mock_response += """
• **Immediate attention required**: Your model shows significant bias that needs urgent addressing
• **Data rebalancing**: Implement stratified sampling to ensure equal representation across sensitive groups
• **Feature engineering**: Review and potentially remove or transform features that correlate strongly with sensitive attributes
• **Ensemble methods**: Consider using ensemble approaches that can reduce individual model biases
• **Bias-aware algorithms**: Implement fairness-constrained optimization during training
"""
        elif bias_score > 25:
            mock_response += """
• **Moderate bias detected**: Apply targeted mitigation strategies
• **Cross-validation with fairness metrics**: Ensure your validation strategy includes fairness evaluation
• **Feature importance analysis**: Investigate which features contribute most to bias
• **Threshold optimization**: Adjust decision thresholds for different groups to achieve fairness
• **Regular bias auditing**: Implement systematic bias checking in your ML pipeline
"""
        else:
            mock_response += """
• **Good fairness level**: Continue current practices with minor improvements
• **Preventive monitoring**: Establish continuous monitoring to prevent bias drift
• **Documentation**: Document your fairness practices for reproducibility
• **Stakeholder engagement**: Include diverse perspectives in model evaluation
• **Incremental improvements**: Fine-tune model parameters for marginal fairness gains
"""
        
        mock_response += f"""

## 2. Model Retraining Approaches

• **Algorithm selection**: For {model_type} models, consider fairness-aware variants like Fair Random Forest or Fairness-constrained SVM
• **Regularization techniques**: Add fairness regularization terms to your loss function
• **Multi-objective optimization**: Balance accuracy and fairness using Pareto optimization
• **Transfer learning**: Use pre-trained models that have been debiased on large, diverse datasets
• **Hyperparameter tuning**: Include fairness metrics in your hyperparameter optimization process

## 3. Data Collection Improvements

"""
        
        if sensitive_features:
            mock_response += f"""
• **Sensitive attribute handling**: For detected sensitive features ({', '.join(sensitive_features[:3])}), ensure balanced representation
• **Data augmentation**: Generate synthetic samples for underrepresented groups using techniques like SMOTE
• **Collection protocol review**: Audit your data collection process for potential sources of bias
• **Diverse data sources**: Expand data collection to include more diverse populations
• **Quality over quantity**: Focus on high-quality, representative samples rather than just volume
"""
        else:
            mock_response += """
• **Proactive sensitive attribute identification**: Implement systematic checks for potential sensitive attributes
• **Diverse sampling strategies**: Use stratified sampling across demographic groups
• **Bias testing in collection**: Test data collection methods for systematic biases
• **Community engagement**: Involve affected communities in data collection design
• **Temporal bias checking**: Ensure data represents current populations, not historical biases
"""
        
        mock_response += """

## 4. Monitoring and Governance Practices

• **Continuous monitoring dashboard**: Implement real-time bias monitoring in production
• **Automated alerts**: Set up alerts when bias metrics exceed acceptable thresholds
• **Regular bias audits**: Schedule quarterly comprehensive bias assessments
• **Version control for fairness**: Track fairness metrics across model versions
• **Stakeholder review process**: Establish regular reviews with diverse stakeholders
• **Incident response plan**: Create procedures for addressing bias issues in production

## 5. Regulatory Compliance Considerations

• **GDPR compliance**: Ensure your model meets EU data protection and algorithmic accountability requirements
• **Equal Credit Opportunity Act**: If applicable, verify compliance with anti-discrimination lending laws
• **Fair Housing Act**: For housing-related models, ensure compliance with fair housing regulations
• **Employment law compliance**: For HR applications, verify compliance with equal employment opportunity laws
• **Documentation requirements**: Maintain detailed records of bias testing and mitigation efforts
• **Explainability standards**: Implement model explainability features for regulatory transparency

## Next Steps Priority

1. **Immediate** (this week): Implement data rebalancing for highest-bias sensitive attributes
2. **Short-term** (1-2 weeks): Set up continuous bias monitoring
3. **Medium-term** (1 month): Retrain model with fairness constraints
4. **Long-term** (ongoing): Establish governance framework and regular auditing

**Confidence Score**: 85% - Recommendations based on standard fairness best practices and your specific bias profile.
"""
        
        return mock_response
    
    def _parse_ai_response(self, ai_response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into structured recommendations"""
        
        # This is a simplified parser - in production, you'd use more sophisticated NLP
        lines = ai_response.split('\n')
        
        structured = {
            "bias_reduction_strategies": [],
            "model_retraining_approaches": [],
            "data_collection_improvements": [],
            "monitoring_practices": [],
            "compliance_considerations": [],
            "confidence_score": 0.8  # Default confidence
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            # Detect sections
            if "bias reduction" in line.lower():
                current_section = "bias_reduction_strategies"
                continue
            elif "retraining" in line.lower() or "model" in line.lower():
                current_section = "model_retraining_approaches"
                continue
            elif "data collection" in line.lower():
                current_section = "data_collection_improvements"
                continue
            elif "monitoring" in line.lower() or "governance" in line.lower():
                current_section = "monitoring_practices"
                continue
            elif "compliance" in line.lower() or "regulatory" in line.lower():
                current_section = "compliance_considerations"
                continue
            elif "confidence" in line.lower():
                # Try to extract confidence score
                import re
                confidence_match = re.search(r'(\d+)%', line)
                if confidence_match:
                    structured["confidence_score"] = float(confidence_match.group(1)) / 100
                continue
            
            # Add bullet points to current section
            if current_section and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                clean_line = line[1:].strip()
                if clean_line:
                    structured[current_section].append(clean_line)
        
        # Ensure each section has at least some content
        for key in ["bias_reduction_strategies", "model_retraining_approaches", 
                   "data_collection_improvements", "monitoring_practices", 
                   "compliance_considerations"]:
            if not structured[key]:
                structured[key] = [f"Review {key.replace('_', ' ')} based on your specific use case"]
        
        return structured
    
    def get_recommendation(self, job_id: str) -> AIRecommendation:
        """Get AI recommendation by job ID"""
        if job_id not in self.ai_recommendations:
            raise Exception("AI recommendation not found")
        return self.ai_recommendations[job_id]


# Add numpy import for the mock function
import numpy as np

# Global AI recommendation service instance
ai_recommendation_service = AIRecommendationService()
