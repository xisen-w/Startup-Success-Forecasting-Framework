from agents.market_agent import MarketAgent
from agents.founder_agent import FounderAgent
from agents.product_agent import ProductAgent
from agents.vc_scout_agent import VCScoutAgent
from agents.integration_agent import IntegrationAgent

def analyze_startup(startup_info, mode="advanced"):
    market_agent = MarketAgent()
    founder_agent = FounderAgent()
    product_agent = ProductAgent()
    vc_scout_agent = VCScoutAgent()
    integration_agent = IntegrationAgent()

    market_analysis = market_agent.analyze(startup_info, mode)
    founder_analysis = founder_agent.analyze(startup_info, mode)
    product_analysis = product_agent.analyze(startup_info, mode)
    vc_evaluation = vc_scout_agent.evaluate(startup_info, mode)

    all_analyses = {
        "Market Analysis": market_analysis,
        "Founder Analysis": founder_analysis,
        "Product Analysis": product_analysis,
        "VC Scout Evaluation": vc_evaluation
    }

    integrated_analysis = integration_agent.integrate_analyses(all_analyses)

    return {
        "Individual Analyses": all_analyses,
        "Integrated Analysis": integrated_analysis
    }

# Usage
startup_info = {
    "description": "AI-powered education platform",
    "market_size": "$2.5 billion",
    "founder_backgrounds": "Oxford University, AI Research experience",
    "product_description": "Personalized learning paths using advanced AI algorithms",
    "key_features": "Adaptive learning, real-time feedback, progress tracking",
    "tech_stack": "Python, TensorFlow, React, AWS",
    "usp": "10x faster learning through AI-optimized content delivery",
    # Add other relevant information
}

result = analyze_startup(startup_info)
print(result)