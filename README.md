# Startup Success Forecasting Framework (SSFF)

The Startup Success Forecasting Framework (SSFF) is a pioneering approach designed to automate the evaluation of startup success potential. Leveraging a blend of traditional machine learning models, Large Language Models (LLMs), and real-time market data analysis, SSFF aims to transform the landscape of venture capital investment by providing deep, actionable insights into the viability of early-stage startups.

## What This Repo Includes

- **Data Exploration Files:** Scripts and notebooks for exploring and understanding the dataset used in model training.
- **Model Training Files:** Code for training the SSFF's underlying machine learning models, including random forest classifiers and neural networks.
- **Pipeline Building Files:** Implementation of the SSFF pipeline, integrating LLM analysis, feature extraction, and prediction models for comprehensive startup evaluation.

## To-Do List

- [ ] Launch a Demo Interface
- [ ] Upgrade to Version 2

## How to Run

To execute the main pipeline and analyze a startup, use the following command:

```bash
python overallPipeline.py
```
### This framework supports two modes of operation:

Simple Mode: Provides a quick evaluation based on predefined criteria.
Advanced Mode: Offers an in-depth analysis incorporating external market data, founder-level segmentation, and custom LLM prompts for nuanced insights.

## Reference Paper

For a detailed understanding of the methodologies and technologies underpinning SSFF, refer to our accompanying paper titled "An Automated Startup Evaluation Pipeline: Startup Success Forecasting Framework (SSFF)". This paper discusses the challenges of early-stage startup evaluation and presents our novel framework as a solution that combines machine learning, natural language processing, and real-time data analysis.

## Key Highlights from the Paper:
Startup Evaluation Pipeline: Exploration of traditional and emerging approaches to startup evaluation, emphasizing the integration of qualitative assessments.
LLM Agent and Prompting Techniques: Insight into how Large Language Models and sophisticated prompting techniques can enhance the evaluation process.
Founder Level Segmentation: A novel approach to categorizing founders based on their experience and potential impact on startup success.
External Knowledge Block: Utilization of real-time market data to enrich the analysis and provide a current view of market conditions and trends.

## Conclusion

The SSFF represents a significant step forward in automating and enhancing the accuracy of startup success evaluations. By combining cutting-edge AI technologies with a deep understanding of the startup ecosystem, SSFF empowers investors, entrepreneurs, and researchers with a powerful tool for making informed decisions.

## Acknowledgments

This work was supported by contributions from Xisen Wang at the University of Oxford under his internship at Vela Partners, where Yigit is his supervisor. Their invaluable insights and expertise have been instrumental in the development of the SSFF.

