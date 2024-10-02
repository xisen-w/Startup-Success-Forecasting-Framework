# Startup Success Forecasting Framework (SSFF)

The Startup Success Forecasting Framework (SSFF) is a pioneering approach designed to automate the evaluation of startup success potential. Leveraging a blend of traditional machine learning models, Large Language Models (LLMs), and real-time market data analysis, SSFF aims to transform the landscape of venture capital investment by providing deep, actionable insights into the viability of early-stage startups.

Link to Paper: https://arxiv.org/abs/2405.19456

## Project Structure

```
project_root/
│
├── models/
│   ├── neural_network.keras
│   ├── random_forest_classifier.joblib
│   └── trained_encoder_RF.joblib
│
├── EDA/
│   └── (exploratory data analysis scripts)
│
├── algorithms/
│   ├── embedding.py
│   ├── similarity.py
│   └── preprocessing.py
│
├── agents/
│   ├── base_agent.py
│   ├── market_agent.py
│   ├── founder_agent.py
│   ├── product_agent.py
│   ├── vc_scout_agent.py
│   └── integration_agent.py
│
├── utils/
│   ├── config.py
│   └── api_wrapper.py
│
├── main.py
└── README.md
```

## Environment Setup

To set up the environment for this project, follow these steps:

1. Ensure you have Python 3.7+ installed on your system.

2. Clone the repository:
   ```
   git clone https://github.com/your-username/Startup-Success-Forecasting-Framework.git
   cd Startup-Success-Forecasting-Framework
   ```

3. Create a virtual environment:
   ```
   python -m venv myenv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source myenv/bin/activate
     ```

5. Install the required packages:
   ```
   pip install openai python-dotenv pydantic google-search-results
   ```

6. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```

7. To deactivate the virtual environment when you're done:
   ```
   deactivate
   ```

## What This Repo Includes

- **Data Exploration Files:** Scripts and notebooks for exploring and understanding the dataset used in model training.
- **Model Training Files:** Code for training the SSFF's underlying machine learning models, including random forest classifiers and neural networks.
- **Pipeline Building Files:** Implementation of the SSFF pipeline, integrating LLM analysis, feature extraction, and prediction models for comprehensive startup evaluation.

![SSFF Architecture](https://github.com/Xisen-Wang/Startup-Success-Forecasting-Framework/assets/118058822/583db157-5293-4006-8bde-4f7ad7e0d3ea)

## Updates

A web interface has been developed! Very easy to interact with: 

<img width="1509" alt="Web Interface Screenshot" src="https://github.com/Xisen-Wang/Startup-Success-Forecasting-Framework/assets/118058822/477be7cf-6884-40d4-ab88-832f63702965">

## To-Do List

- [x] Launch a Demo Interface
- [x] Upgrade to Version 2

## How to Run

To execute the main pipeline and analyze a startup, use the following command:

```bash
python main.py
```

### This framework supports two modes of operation:

- Simple Mode: Provides a quick evaluation based on predefined criteria.
- Advanced Mode: Offers an in-depth analysis incorporating external market data, founder-level segmentation, and custom LLM prompts for nuanced insights.

## Reference Paper

For a detailed understanding of the methodologies and technologies underpinning SSFF, refer to our accompanying paper titled "An Automated Startup Evaluation Pipeline: Startup Success Forecasting Framework (SSFF)". This paper discusses the challenges of early-stage startup evaluation and presents our novel framework as a solution that combines machine learning, natural language processing, and real-time data analysis.

## Key Highlights from the Paper:
- Startup Evaluation Pipeline: Exploration of traditional and emerging approaches to startup evaluation, emphasizing the integration of qualitative assessments.
- LLM Agent and Prompting Techniques: Insight into how Large Language Models and sophisticated prompting techniques can enhance the evaluation process.
- Founder Level Segmentation: A novel approach to categorizing founders based on their experience and potential impact on startup success.
- External Knowledge Block: Utilization of real-time market data to enrich the analysis and provide a current view of market conditions and trends.

## Conclusion

The SSFF represents a significant step forward in automating and enhancing the accuracy of startup success evaluations. By combining cutting-edge AI technologies with a deep understanding of the startup ecosystem, SSFF empowers investors, entrepreneurs, and researchers with a powerful tool for making informed decisions.

## Acknowledgments

This work was supported by contributions from Xisen Wang at the University of Oxford under his internship at Vela Partners, where Yigit is his supervisor. Their invaluable insights and expertise have been instrumental in the development of the SSFF.
