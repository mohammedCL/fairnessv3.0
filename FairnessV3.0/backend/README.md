# AI Fairness Assessment Platform - Backend

A comprehensive API for AI model fairness assessment with bias detection, mitigation strategies, and AI-powered recommendations.

## Features

### 🔍 **6-Step Fairness Assessment Workflow**

1. **Model & Dataset Upload** - Support for multiple formats (.pkl, .joblib, .onnx, .csv, .json, .parquet)
2. **Model Analysis & Sensitive Feature Detection** - Automatic detection of model type and sensitive attributes
3. **Bias Detection** - Three types: Direct, Proxy, and Model bias
4. **Fairness Scoring & Visualization** - Comprehensive metrics and charts
5. **Bias Mitigation** - Preprocessing, In-processing, and Post-processing strategies
6. **AI-Powered Recommendations** - LLM-generated actionable insights

### 🎯 **Bias Detection Capabilities**

- **Direct Bias**: Explicit use of protected attributes
- **Proxy Bias**: Correlated features acting as proxies
- **Model Bias**: Prediction disparities across demographic groups

### 📊 **Fairness Metrics**

- Demographic Parity Difference
- Disparate Impact Ratio
- Equalized Odds Difference
- Equal Opportunity Difference
- Statistical Parity
- Calibration metrics

### 🛠️ **Mitigation Strategies**

- **Preprocessing**: Data reweighing, disparate impact remover, data augmentation
- **In-processing**: Fairness constraints, regularization, adversarial debiasing
- **Post-processing**: Threshold optimization, calibration adjustment

### 🤖 **AI-Powered Recommendations**

- Integration with OpenAI GPT-4 and Anthropic Claude
- Personalized bias reduction strategies
- Model retraining approaches
- Data collection improvements
- Compliance considerations

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate to backend**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional for basic functionality)
   ```

4. **Run the server**:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Upload
- `POST /api/upload/model` - Upload model file
- `POST /api/upload/train-dataset` - Upload training dataset
- `POST /api/upload/test-dataset` - Upload test dataset
- `GET /api/upload/status/{upload_id}` - Get upload status

### Analysis
- `POST /api/analysis/start` - Start bias analysis
- `GET /api/analysis/job/{job_id}` - Get analysis job status
- `GET /api/analysis/results/{job_id}` - Get analysis results
- `GET /api/analysis/bias-metrics/{job_id}` - Get bias metrics
- `GET /api/analysis/fairness-score/{job_id}` - Get fairness score
- `GET /api/analysis/visualizations/{job_id}` - Get visualization charts

### Mitigation
- `POST /api/mitigation/start` - Start mitigation job
- `GET /api/mitigation/job/{job_id}` - Get mitigation job status
- `GET /api/mitigation/results/{job_id}` - Get mitigation results
- `GET /api/mitigation/comparison/{job_id}` - Get before/after comparison
- `GET /api/mitigation/strategies` - Get available strategies

### AI Recommendations
- `POST /api/ai/generate` - Generate AI recommendations
- `GET /api/ai/recommendation/{job_id}` - Get AI recommendation
- `POST /api/ai/chat` - Chat follow-up questions
- `GET /api/ai/status` - Get AI service status

## Usage Example

### 1. Upload Files

```python
import requests

# Upload model
with open('model.pkl', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload/model', 
                           files={'file': f})
    model_upload_id = response.json()['upload_id']

# Upload dataset
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload/train-dataset', 
                           files={'file': f})
    dataset_upload_id = response.json()['upload_id']
```

### 2. Start Analysis

```python
response = requests.post('http://localhost:8000/api/analysis/start', 
                        params={
                            'model_upload_id': model_upload_id,
                            'train_dataset_upload_id': dataset_upload_id
                        })
analysis_job_id = response.json()['job_id']
```

### 3. Get Results

```python
# Check analysis status
response = requests.get(f'http://localhost:8000/api/analysis/job/{analysis_job_id}')
status = response.json()['status']

# Get results when completed
if status == 'completed':
    response = requests.get(f'http://localhost:8000/api/analysis/results/{analysis_job_id}')
    results = response.json()
```

### 4. Apply Mitigation

```python
response = requests.post('http://localhost:8000/api/mitigation/start',
                        params={
                            'analysis_job_id': analysis_job_id,
                            'strategy': 'preprocessing'
                        })
mitigation_job_id = response.json()['job_id']
```

### 5. Get AI Recommendations

```python
response = requests.post('http://localhost:8000/api/ai/generate',
                        json={
                            'analysis_job_id': analysis_job_id,
                            'mitigation_job_id': mitigation_job_id
                        })
recommendations = response.json()
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for GPT-4 recommendations
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude recommendations
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: false)

### AI Recommendations

The system works with mock responses by default. To enable full AI features:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/)
2. Set the environment variable in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   # OR
   ANTHROPIC_API_KEY=your_key_here
   ```

## Supported Formats

### Models
- **Pickle** (.pkl) - scikit-learn, XGBoost, LightGBM models
- **Joblib** (.joblib) - scikit-learn models
- **ONNX** (.onnx) - Cross-platform models

### Datasets
- **CSV** (.csv) - Comma-separated values
- **JSON** (.json) - JSON format
- **Parquet** (.parquet) - Apache Parquet
- **Excel** (.xlsx) - Microsoft Excel

## Architecture

```
backend/
├── app/
│   ├── api/                 # API route handlers
│   │   ├── upload.py       # File upload endpoints
│   │   ├── analysis.py     # Bias analysis endpoints
│   │   ├── mitigation.py   # Mitigation endpoints
│   │   └── ai_recommendations.py  # AI endpoints
│   ├── services/           # Business logic
│   │   ├── upload_service.py
│   │   ├── analysis_service.py
│   │   ├── mitigation_service.py
│   │   └── ai_recommendation_service.py
│   ├── models/             # Data models
│   │   └── schemas.py      # Pydantic models
│   └── utils/              # Utilities
│       └── file_validators.py
├── uploads/                # File storage
├── main.py                 # FastAPI application
└── requirements.txt        # Dependencies
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black .

# Check types
mypy .

# Lint
flake8 .
```

### Adding New Features

1. Define data models in `app/models/schemas.py`
2. Implement business logic in `app/services/`
3. Create API endpoints in `app/api/`
4. Add tests in `tests/`

## Production Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

- Set production environment variables
- Configure secure file storage (S3, etc.)
- Set up proper logging and monitoring
- Enable HTTPS with reverse proxy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the example usage above
