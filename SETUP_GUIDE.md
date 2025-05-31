# SSFF Setup Guide

## Missing Components & Setup Instructions

### 1. 🔑 API Keys Setup (CRITICAL)

Create a `.env` file in the project root with your API keys:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here

# Model Configuration
DEFAULT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
```

**How to get these keys:**
- **OpenAI API Key**: Sign up at https://platform.openai.com/api-keys
- **SerpAPI Key**: Sign up at https://serpapi.com/manage-api-key

### 2. 📊 Missing Data Files (CRITICAL)

The framework expects these CSV files containing startup founder data:

```
data/
├── Successful/
│   └── successful_profiles.csv
├── Unsuccessful/
│   └── unsuccessful_profiles.csv
├── successful/
│   └── segmented_successful_profiles.csv
└── unsuccessful/
    └── segmented_unsuccessful_profiles.csv
```

**Expected CSV Schema:**
- Founder background text
- Company description
- Success/failure outcome (0/1)
- Segmentation level (L1-L5)

### 3. 🚀 Quick Test Run

To test if everything works:

```bash
# Test the web interface
streamlit run app.py

# Test the main pipeline
python overallPipeline.py
```

### 4. 📈 Model Performance Context

The framework uses:
- **Founder Segmentation**: L1 (24% success) → L5 (92% success)
- **Multi-Agent Analysis**: Market, Product, Founder, Integration
- **ML Models**: Neural Networks + Random Forest
- **Real-time Data**: Market analysis via SerpAPI

### 5. 🛠 Path Fixes Applied

- Created proper data directory structure
- All dependencies installed
- Models are present and ready

### 6. 📚 Data Sources Needed

You'll need to obtain or create:
- Founder profile datasets with success/failure labels
- Segmented founder data (processed through LLM classification)
- Historical startup outcome data

### 7. 🎯 Current Status

✅ **Working**: Models, dependencies, code structure
❌ **Missing**: API keys, training data
⚠️ **Partial**: Can run without data but needs API keys for full functionality

### 8. 💡 Recommendation

1. **Immediate**: Set up API keys to enable LLM functionality
2. **Short-term**: Source or create sample founder datasets
3. **Long-term**: Collect comprehensive startup outcome data for model training 