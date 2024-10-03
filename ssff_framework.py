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


if __name__ == "__main__":
    from utils.api_wrapper import OpenAIAPI
    from agents.market_agent import MarketAgent
    from agents.founder_agent import FounderAgent
    from agents.product_agent import ProductAgent
    from agents.vc_scout_agent import VCScoutAgent
    from agents.integration_agent import IntegrationAgent
    from pydantic import BaseModel

    # Test OpenAIAPI
    print("Testing OpenAIAPI...")
    api = OpenAIAPI(model_name="gpt-3.5-turbo")
    completion = api.get_completion(
        system_content="You are a helpful assistant.",
        user_content="What is the capital of France?"
    )
    print(f"OpenAI Completion: {completion}")

    # Test structured output
    class TestSchema(BaseModel):
        summary: str
        score: int

    structured_output = api.get_structured_output(
        schema_class=TestSchema,
        user_prompt="Summarize the benefits of exercise and give it a score out of 10.",
        system_prompt="You are a health expert. Provide a summary and a score for the given topic."
    )
    print(f"Structured Output: {structured_output}")

    # Test Agents
    print("\nTesting Agents...")
    startup_info = {
        "description": "AI-powered education platform",
        "market_size": "$2.5 billion",
        "founder_backgrounds": "Oxford University, AI Research experience",
        "product_description": "Personalized learning paths using advanced AI algorithms",
        "key_features": "Adaptive learning, real-time feedback, progress tracking",
        "tech_stack": "Python, TensorFlow, React, AWS",
        "usp": "10x faster learning through AI-optimized content delivery",
    }

    market_agent = MarketAgent()
    founder_agent = FounderAgent()
    product_agent = ProductAgent()
    vc_scout_agent = VCScoutAgent()
    integration_agent = IntegrationAgent()

    print("Market Agent Analysis:")
    market_analysis = market_agent.analyze(startup_info, mode="advanced")
    print(market_analysis)

    print("\nFounder Agent Analysis:")
    founder_analysis = founder_agent.analyze(startup_info, mode="advanced")
    print(founder_analysis)

    print("\nProduct Agent Analysis:")
    product_analysis = product_agent.analyze(startup_info, mode="advanced")
    print(product_analysis)

    print("\nVC Scout Evaluation:")
    vc_evaluation = vc_scout_agent.evaluate(startup_info, mode="advanced")
    print(vc_evaluation)

    print("\nIntegration Agent Analysis:")
    all_analyses = {
        "Market Analysis": market_analysis,
        "Founder Analysis": founder_analysis,
        "Product Analysis": product_analysis,
        "VC Scout Evaluation": vc_evaluation
    }
    integrated_analysis = integration_agent.integrate_analyses(all_analyses)
    print(integrated_analysis)

    print("\nAll tests completed.")