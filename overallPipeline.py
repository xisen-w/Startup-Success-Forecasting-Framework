import os
import pandas as pd
from openai import OpenAI
import json
from serpapi import GoogleSearch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

##The main function is here.
## This is used for the Overall Analysis Agent - Xisen

# Manage all the API_Keys here
#OPENAI_API_KEY = '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'
OPENAI_API_KEY = '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'
SERP_API_KEY = "a166ed8b1f69417d2a2466ea067756eb18f4d09ac0e2983cca254a75e7ca5367"

# Put some universal variables here

Market_Report = ""
News_Report = ""

# Load all models here

model_path = os.getcwd() + "/models"
neuralNetworks_name = "neural_network.keras"
Random_Forest_name = "random_forest_classifier.joblib"
model_neural_network = load_model(os.path.join(model_path, neuralNetworks_name))
model_random_forest = joblib.load(os.path.join(model_path, Random_Forest_name))

# This is the key analysis function
def mainPipeline(startup_description, founder_description, mode, model):
    """
    :param startup_info: [description], mode: simple, advanced
    :return: various analyses & the main pipeline
    """
    #prediction = sideVCScout(test_startup, model)
    #print(prediction)

    # Combine the information together
    startup_info = "Startup Description:" + startup_description + "Founder Information: " + founder_description

    # Get prediction
    prediction = sideVCScout(startup_info, 'gpt-3.5-turbo')

    # Determine if I need to segment the startup_info myself
    if input("Have you preprocessed it?") == "no":
        print("ANALYZING--------------------------------")
        startup_info = startup_info_to_JSON(startup_info, model)

        print(startup_info)
        print(type(startup_info))

    # Let the agents do their respective analyses
    print("MARKET INFO ANALYZING--------------------------------")
    market_info = market_analysis(startup_info, mode, model)
    print(market_info)
    print("PRODUCT INFO ANALYZING--------------------------------")
    product_info = product_analysis(startup_info, mode, model)
    print(product_info)
    print("FOUNDER INFO ANALYZING--------------------------------")
    founder_info = founder_analysis(startup_info, mode, model)
    print(founder_info)
    print("FINAL DECISION ANALYZING--------------------------------")

    # # Integration Process
    if mode == "simple":
        final_decision = integrate_analyses(market_info, product_info, founder_info, prediction, mode)

    elif mode == "advanced":
        # Do Segmentation
        Founder_Segmentation = LevelSegmentation(founder_description, model)
        print("Segmention is: ", Founder_Segmentation)

        # Get Fit Score
        Founder_Idea_Fit = getFit(founder_description, startup_description)
        print("Idea fit is: ", Founder_Idea_Fit)

        # Merge In
        founder_info += f"After modelling, the segmentaton of the founder is {Founder_Segmentation}, with L1 being least likely to be successful and L5 being most likely to be successful. L5 founders are 3.8 times more likely to succeed than L1 founders. Take this into account ."
        founder_info += f"The Founder_Idea_Fit Score of this startup is measured to be {Founder_Idea_Fit}. The score ranges from -1 to 1, with 1 being that the startup fits with the founder's background well, and -1 being the least fit. Also take this into account in your report.  "

        final_decision = integrate_analyses(market_info, product_info, founder_info, prediction, mode)

        print(final_decision)
        return [final_decision, market_info, product_info, founder_info, Market_Report, News_Report, Founder_Segmentation, Founder_Idea_Fit, prediction]

    else:
        return None
    return [final_decision, market_info, product_info, founder_info, "N/A", "N/A", "N/A", "N/A", "N/A"]

def getResponse(System_Content, User_Content, input_model):
    client = OpenAI(api_key=OPENAI_API_KEY, )
    completion = client.chat.completions.create(
        model=input_model,
        messages=[
            {"role": "system", "content": System_Content},
            {"role": "user", "content": User_Content}
        ]
    )
    return completion.choices[0].message.content

def getGoogleResults(Keywords):
    params = {
        "api_key": SERP_API_KEY,
        "engine": "google",
        "q": Keywords,
        "google_domain": "google.com",
        "hl": "en"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def getJSONResponse(System_Content, User_Content, input_model):
    client = OpenAI(api_key=OPENAI_API_KEY, )
    completion = client.chat.completions.create(
        model=input_model,
        messages=[
            {"role": "system", "content": System_Content},
            {"role": "user", "content": User_Content}
        ],
        response_format = {"type": "json_object"}
    )
    return completion.choices[0].message.content

def market_analysis(startup_info, mode, model):
    analysis_prompt = (
        "You are an experienced market analyst equipped with comprehensive knowledge of "
        "various industry sectors and startup ecosystems. Your task is to analyze a startup "
        "based on the following information: {description}\n\n"
        "Consider the current market size, projected growth rate, existing competition, "
        "and prevailing market trends. Discuss the startup's go-to-market strategies "
        "and whether the timing for its entry into the market is favorable or not. "
        "Reflect on any historical successes or failures in this market that could be "
        "indicative of the startup's potential future performance. Provide a well-reasoned "
        "analysis and conclude with a market viability score from 1 to 10. Here is what you "
        "know about the startup:\n\n"
        "- Description: {description}\n"
        "- Market_size: {market_size}\n"
        "- Competition: {competition}\n"
        "- Market Growth Rate: {growth_rate}\n"
        "- Go_to_market_strategy: {go_to_market_strategy}\n\n"
        "- Market Trends: {market_trends}\n"
        "Begin your analysis by addressing the timing of market entry, taking into account "
        "the startup's unique position and how similar companies have fared in recent years."
        "Please think and analyze step by step and take a deep breath."
    ).format(
        description=startup_info.get('description', ""),
        go_to_market_strategy=startup_info.get('go_to_market_strategy', ""),
        competition=startup_info.get('competition', ""),
        market_size=startup_info.get('market_size', ""),
        growth_rate=startup_info.get('growth_rate', ""),
        market_trends=startup_info.get('market_trends', "")
    )
    if mode == "simple":
        pass

    if mode == "advanced":
        # first get external knowledge through searching from APIs
        external_knowledge = externalMarketKnowledge(startup_info.get('description'))
        external_prompt = f"Consider the external market knowledge of {startup_info['name'], external_knowledge}."
        analysis_prompt += f"\n{external_prompt}"
        global Market_Report
        Market_Report = external_knowledge
    return getResponse(analysis_prompt, "Please think and analyze step by step and take a deep breath", model)


def product_analysis(startup_info, mode, model):
    analysis_prompt = (
        "You're a seasoned product analyst known for your critical evaluations of new technology products. "
        "Today, you're examining a startup's offering with the details provided below:\n\n"
        "Assess the product's market fit, innovation level, scalability potential, and user reception. "
        "Investigate the technology behind the product, its differentiators, and any existing user feedback. "
        "Also, review user ratings and feedback on sites such as ProductHunt and G2Crowd to gauge market reception. "
        "Summarize your findings by providing a product viability score from 1 to 10, taking into account "
        "web traffic growth as a metric for user interest and market traction.\n\n"
        "- Product Details: {product_details}\n"
        "- Technology Stack: {technology_stack}\n"
        "- Scalability: {scalability}\n"
        "- User Feedback: {user_feedback}\n\n"
        "- Product Market-fit: {product_fit}\n\n"
        "Please provide a well-reasoned product analysis incorporating the above aspects."
        "Please think and analyze step by step and take a deep breath."
    ).format(
        product_details=startup_info.get('product_details'),
        technology_stack=startup_info.get('technology_stack', ''),
        product_fit=startup_info.get('product_fit', 'growing steadily'),
        user_feedback=startup_info.get('user_feedback', 'positive overall'),
        scalability=startup_info.get('scalability', '')
    )
    if mode == "simple":
        pass

    if mode == "advanced":
        # first get external knowledge through searching from APIs
        external_knowledge = externalProductKnowledge(startup_info, 10)
        external_prompt = f"Note the information below might depict companies of the same name but of different entities. Ignore those that are not the same entity with the company of our original analysis. Consider the latest product/company knowledge of {startup_info['name']}:\n {external_knowledge}."
        analysis_prompt += f"\n{external_prompt}"
        global News_Report
        News_Report = News_Report
    return getResponse(analysis_prompt, "Please think and analyze step by step and take a deep breath.", model)


def founder_analysis(startup_info, mode, model):
    analysis_prompt = (
        "As a highly qualified analyst specializing in startup founder assessment, you've been tasked "
        "with evaluating the founding team of a new company. Here's what you need to know:\n\n"
        "Consider the founders' educational background, industry experience, leadership capabilities, "
        "and their ability to align and execute on the company's vision. Look into their digital footprint "
        "on platforms like Twitter and LinkedIn, and any available data from the Crunchbase and Diffbot APIs "
        "to enrich your analysis. Score the founders' competency on a scale of 1 to 10, and provide insights "
        "into their strengths and potential challenges.\n\n"
        "- Founders' Backgrounds: {founder_backgrounds}\n"
        "- Track' Records: {track_records}\n"
        "- leadership Skills: {leadership_skills}\n"
        "- Vision and Alignment: {vision_alignment}\n"
        "- Team Dynamics: {team_dynamics}\n\n"
        "Your evaluation should culminate in a comprehensive founders' competency assessment."
        "Please think and analyze step by step and take a deep breath."
    ).format(
        team_dynamics=startup_info['team_dynamics'],
        founder_backgrounds=startup_info.get('founder_backgrounds', ''),
        leadership_skills=startup_info.get('leadership_skills', ''),
        vision_alignment=startup_info.get('vision_alignment', ''),
        track_records=startup_info.get('track_records', "")
    )
    if mode == "simple":
        pass

    return (getResponse(analysis_prompt, "Please think and analyze step by step and take a deep breath.", model))


def integrate_analyses(market_info, product_info, founder_info, prediction, mode):
    prompt = """
    "Imagine you are the chief analyst at a venture capital firm, tasked with integrating the analyses of three specialized teams to provide a comprehensive investment insight. Below are detailed examples to guide your analysis. Your output should be similarly structured but in much greater detail: 
    Score each from 1 to 10 (10 is the best & most competitive). Specify the score to 2 digits and give very strong justification for it.
    
    Example 1:
    Market Viability: 8.23/10 - The market is on the cusp of a regulatory shift that could open up new demand channels, supported by consumer trends favoring sustainability. Despite the overall growth, regulatory uncertainty poses a potential risk.
    Product Viability: 7.36/10 - The product introduces an innovative use of AI in renewable energy management, which is patent-pending. However, it faces competition from established players with deeper market penetration and brand recognition.
    Founder Competency: 9.1/10 - The founding team comprises industry veterans with prior successful exits and a strong network in the energy sector. Their track record includes scaling similar startups and navigating complex regulatory landscapes.
    
    Recommendation: Invest. The team's deep industry expertise and innovative product position it well to capitalize on the market's regulatory changes. Although competition is stiff, the founders' experience and network provide a competitive edge crucial for market adoption and navigating potential regulatory hurdles.
    
    Example 2:
    Market Viability: 5.31/10 - The market for wearable tech is saturated, with slow growth projections. However, there exists a niche but growing interest in wearables for pet health.
    Product Viability: 6.5/10 - The startup's product offers real-time health monitoring for pets, a feature not widely available in the current market. Yet, the product faces challenges with high production costs and consumer skepticism about the necessity of such a device.
    Founder Competency: 6.39/10 - The founding team includes passionate pet lovers with backgrounds in veterinary science and tech development. While they possess the technical skills and passion for the project, their lack of business and scaling experience is a concern.
    
    Recommendation: Hold. The unique product offering taps into an emerging market niche, presenting a potential opportunity. However, the combination of a saturated broader market, challenges in justifying the product's value to consumers, and the team's limited experience in business management suggests waiting for clearer signs of product-market fit and strategic direction.
    
    Take a deep breath and analyze step by step. Your team has presented you with the following scores & analyses for a new startup:
    
    """
    user_prompt = (
        "Imagine you are the chief analyst at a venture capital firm, integrating the analyses "
        "of three specialized teams into a cohesive investment insight. Your team has provided you "
        "with the following scores:\n\n"
        "Market Viability: {market_info}\n"
        "Product Viability: {product_info}\n"
        "Founder Competency: {founder_info}\n"
        "Using your expertise, synthesize these scores to present an overall investment recommendation. "
        "State whether you would advise 'Invest' or 'Hold', including a comprehensive rationale for your decision."
        "Please think and analyze step by step and take a deep breath."
        "You are very professional and you generate insightful analysis."
    ).format(
        market_info=market_info,
        product_info=product_info,
        founder_info=founder_info,
    )

    if mode == "advanced":
        prediction_prompt = f"\n In addition, your model has predicted if the startup will success with 65% accuracy. The outcome is {prediction} Do reference this in your analysis, but don't over-rely on this prediction, as sometimes it gets wrong. "
        user_prompt += prediction_prompt

    return getResponse(prompt, user_prompt, "gpt-4")


def startup_info_to_JSON(startup_description, model):
    '''
    Function to use few-shot prompting to convert
    :param model: the use of model, eg. gpt-3.5-turbo
    :param startup_description: a single string
    :return: a nice json
    '''
    prompt = f"""
        Convert the following startup description into a detailed JSON structure. Here are the descriptions for each key:

        - name: The official name of the startup.
        - description: A brief overview of what the startup does.
        - market_size: The size of the market the startup is targeting.
        - growth_rate: The growth rate of the market.
        - competition: Key competitors in the space.
        - market_trends: Current trends within the market.
        - go_to_market_strategy: The startup's plan for entering the market.
        - product_details: Details about the startup's product or service.
        - technology_stack: Technologies used in the product.
        - scalability: How the product can scale.
        - user_feedback: Any feedback received from users.
        - product_fit: How well the product fits the target market.
        - founder_backgrounds: Background information on the founders.
        - track_records: The track records of the founders.
        - leadership_skills: Leadership skills of the team.
        - vision_alignment: How the team's vision aligns with the product.
        - team_dynamics: The dynamics within the startup team.
        - web_traffic_growth: Information on the growth of web traffic to the startup's site.
        - social_media_presence: The startup's presence on social media.
        - investment_rounds: Details of any investment rounds.
        - regulatory_approvals: Any regulatory approvals obtained.
        - patents: Details of any patents held by the startup.

        Example:
        Description: "Startup ABC is revolutionizing the fintech industry with its blockchain-based payment solution. The platform enables instant, secure, and fee-free transactions. Founded by a team of blockchain experts and fintech veterans, Startup ABC aims to disrupt traditional banking services."

        JSON structure:
        {{
            "name": "Startup ABC",
            "description": "Revolutionizing the fintech industry with a blockchain-based payment solution.",
            "market_size": "Estimated at $5 billion",
            "growth_rate": "20% annually",
            "competition": "Traditional banks and other fintech startups",
            "market_trends": "Increasing adoption of blockchain technology",
            "go_to_market_strategy": "Partnerships with e-commerce platforms",
            "product_details": "Instant, secure, and fee-free transactions.",
            "technology_stack": "Blockchain, Cryptography",
            "scalability": "Capable of handling millions of transactions per day",
            "user_feedback": "Highly positive",
            "product_fit": "Fills a significant gap in current payment processing services",
            "founder_backgrounds": "Experts in blockchain and fintech",
            "track_records": "Previously successful fintech ventures",
            "leadership_skills": "Strong, visionary leadership",
            "vision_alignment": "Committed to disrupting traditional banking",
            "team_dynamics": "Collaborative and innovative",
            "web_traffic_growth": "50% month-over-month growth",
            "social_media_presence": "Strong presence on Twitter and LinkedIn",
            "investment_rounds": "Seed round raised $2 million",
            "regulatory_approvals": "Compliant with all applicable fintech regulations",
            "patents": "2 patents on blockchain transaction algorithms"
        }}
        """
    startup_info = getResponse(prompt, f"Now, convert this startup description: {startup_description}", model)
    try:
        startup_info = json.loads(startup_info)
        return startup_info
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return()


def externalMarketKnowledge(startup_info):
    """
    Generates a market knowledge report for a given startup by synthesizing external market data.

    Parameters:
    - startup_info: A string containing information about the startup.

    Returns:
    [A string of kywords, A string containing the synthesized market report.]
    """

    # Synthesize keywords on the market
    Keywords_Prompt = ("You will assist me in finding external market knowledge about a startup. Think step by step. "
                       "Your task is to summarise the information into 1 keyword that best describes the market that the startup is in. "
                       "Sample Output: Chinese Pharmaceutical Market.")
    keywords = getResponse(Keywords_Prompt, startup_info, 'gpt-3.5-turbo')
    keywords += ", Growth, Trend, Size, Revenue"
    print("Keywords: ", keywords)

    # Simulate a fixed keyword for demonstration; replace with the variable 'keywords' in actual use
    #keywords = "Chinese Education Consulting Market"

    # Call the search API
    results = getGoogleResults(keywords)

    # Process organic search results
    organic_results = results["organic_results"][:10]
    organic_knowledge = "".join(
        [result.get('title', '') + result.get('snippet', '') + "\n" for result in organic_results])
    print(organic_knowledge)

    # Initialize brainstorm_results to an empty string
    brainstorm_results = ""

    # Try to process related questions if they exist
    related_QAs = results.get("related_questions", [])
    if related_QAs:
        for related_QA in related_QAs[:]:
            title = related_QA.get('title', "")
            question = related_QA.get('question', "")
            snippet = related_QA.get('snippet', "")
            date = related_QA.get('date', "")
            brainstorm_results += f"{title} + {question} + {snippet} + {date}\n"
    else:
        print("No related questions found.")

    print("BRAINSTORMING---------- ", brainstorm_results)

    # Compile overall knowledge
    Overall_Knowledge = "Here are the web search information about the market:" + organic_knowledge + "\nHere are the contexts:" + brainstorm_results

    # Generate market report
    Conversion_Prompt = ("You will assist me in finding external market knowledge about a startup. "
                         "After google search, you are given important context information and data (most of the time)"
                         "Now please summarise the information as a report about the growth & size of the market, alongside with your existing knowledge"
                         "Also give insights on the timing of entering the market now & also market sentiment. "
                         "Make your response structured and in detail.")
    market_report = getResponse(Conversion_Prompt, Overall_Knowledge, 'gpt-4')

    return [keywords, market_report]


# External Blocks here [For March 14th]

def externalProductKnowledge(startup_info, N):
    # Synthesize keywords on the market
    keywords = startup_info['name']
    keywords += " News"

    # Call the search API
    results = getGoogleResults(keywords)

    # Process organic search results
    organic_results = results["organic_results"][:N]
    organic_knowledge = "".join(
        [result.get('title', '') + result.get('snippet', '') + "\n" for result in organic_results])
    print(organic_knowledge)

    # Initialize brainstorm_results to an empty string
    brainstorm_results = ""

    # Try to process related questions if they exist
    related_QAs = results.get("related_questions", [])
    if related_QAs:
        for related_QA in related_QAs[:]:
            title = related_QA.get('title', "")
            question = related_QA.get('question', "")
            snippet = related_QA.get('snippet', "")
            date = related_QA.get('date', "")
            brainstorm_results += f"Title: {title} + Question: {question} + Snippet: {snippet} + Date: {date}\n"
    else:
        print("No related questions found.")

    related_news = " "
    top_stories = results.get("top_stories", [])
    if top_stories:
        for top_story in top_stories[:]:
            title = top_story.get('title', "")
            source = top_story.get('source', "")
            date = top_story.get('date', "")
            related_news += f"Title: {title} + Source: {source} + Date:{date} + \n"
    else:
        print("No news found.")

    print("FINDING_NEWS---------- ", related_news)

    # Compile overall knowledge
    Overall_Knowledge = "Here are the web search information about the company:" + organic_knowledge + "\nHere are the contexts:" + brainstorm_results + "\nHere are the related news:" + related_news

    # Generate market report
    Conversion_Prompt = ("You will assist me in summarising the latest information and news about the company. "
                         "After google search, you are given important context information and data (most of the time)"
                         "Now please summarise the information as a report to highlight the latest information and public sentiment towards the company and its product, alongside with your existing knowledge"
                         "Make your response structured and in detail.")
    news_report = getResponse(Conversion_Prompt, Overall_Knowledge, 'gpt-4')

    return news_report


def externalFounderKnowledge(startup_info):
    # Synthesize Keywords & Use different APIs
    # Twitter, Bing & etc.
    return "Not Available Right Now."


def sideVCScout(startup_inco, model):
    System_Content = """
    As an analyst specializing in startup evaluation, your task is to categorize startups based on specific criteria related to their market, financial performance, product, team, funding, customer feedback, operational efficiency, and technological innovation. For each of the following questions, please provide a categorical response: 'Yes', 'No', 'N/A', 'Small', 'Medium', 'Large', 'Slower', 'Same', 'Faster', 'Not Adaptable', 'Somewhat Adaptable', 'Very Adaptable', 'Poor', 'Average', 'Excellent', 'Below Average', 'Above Average', 'Decreased', 'Remained Stable', 'Increased', 'Unknown', 'Recognized', 'Highly Regarded', 'Negative', 'Mixed', 'Positive', 'Weak', 'Moderate', 'Strong', 'Less Efficient', 'Efficient', 'More Efficient', 'Higher', 'Same', 'Lower', 'Fragile', 'Adequate', 'Robust', 'Rarely', 'Sometimes', 'Often', 'No', 'Mentioned', 'Emphasized', 'Too Early', 'Just Right', 'Too Late'.

    Think step by step and analyze with reasoning. Think critically and analyze carefully. But do not generate anything other than the JSON itself. 

    Questions:
    "Is the startup operating in an industry experiencing growth? [Yes/No/N/A]",
    "Is the target market size for the startup's product/service considered large? [Small/Medium/Large/N/A]",
    "Does the startup demonstrate a fast pace of development compared to competitors? [Slower/Same/Faster/N/A]",
    "Is the startup considered adaptable to market changes? [Not Adaptable/Somewhat Adaptable/Very Adaptable/N/A]",
    "How would you rate the startup's execution capabilities? [Poor/Average/Excellent/N/A]",
    "Has the startup raised a significant amount of funding in its latest round? [Below Average/Average/Above Average/N/A]",
    "Has the startup's valuation increased with time? [Decreased/Remained Stable/Increased/N/A]",
    "Are well-known investors or venture capital firms backing the startup? [Unknown/Recognized/Highly Regarded/N/A]",
    "Are the reviews and testimonials for the startup predominantly positive? [Negative/Mixed/Positive/N/A]",
    "Do market surveys indicate a strong product-market fit for the startup? [Weak/Moderate/Strong/N/A]",
    "Does the sentiment analysis of founder and company descriptions suggest high positivity? [Negative/Neutral/Positive/N/A]",
    "Are terms related to innovation frequently mentioned in the company's public communications? [Rarely/Sometimes/Often/N/A]",
    "Does the startup mention cutting-edge technology in its descriptions? [No/Mentioned/Emphasized/N/A]",
    "Considering the startup's industry and current market conditions, is the timing for the startup's product or service right? [Too Early/Just Right/Too Late/N/A]"

    Sample JSON Output Format: 
        {
          "startup_analysis_responses": {
            "industry_growth": "No",
            "market_size": "Large",
            "development_pace": "Faster",
            "market_adaptability": "Very Adaptable",
            "execution_capabilities": "Average",
            "funding_amount": "Above Average",
            "valuation_change": "Increased",
            "investor_backing": "Highly Regarded",
            "reviews_testimonials": "Positive",
            "product_market_fit": "Strong",
            "sentiment_analysis": "Negative",
            "innovation_mentions": "Often",
            "cutting_edge_technology": "Emphasized",
            "timing": "Just Right"
          }
        }

    Example provided for context for analysis:
    Reflecting on Startup ABC, which is attempting to leverage AI for healthcare diagnostics, it's essential to provide a balanced view of its situation:
    
    - Industry growth: Yes, but with the caveat that while AI integration in healthcare shows promise, the path is fraught with regulatory and ethical considerations.
    - Market size: Medium. While there's a need for healthcare innovation, the market is segmented and heavily regulated, which could limit scalability.
    - Development pace: Same. The startup's development pace aligns with industry standards, but doesn't necessarily outpace competitors, reflecting the challenges of innovating in a complex field.
    - Adaptability: Somewhat Adaptable. The startup has shown some flexibility, but healthcare's regulatory environment limits rapid pivoting compared to other tech sectors.
    - Execution capabilities: Average. While there have been successes, the startup faces challenges in scaling and integrating into existing healthcare systems.
    - Funding amount: Average. The startup has secured funding, but not at levels that would indicate overwhelming investor confidence, suggesting a cautious optimism.
    - Valuation trend: Remained Stable. The stable valuation indicates steady progress but lacks the explosive growth potential investors might seek in less regulated industries.
    - Investor backing: Recognized. While backed by known investors, the startup has yet to attract the level of high-profile venture capital that signifies industry-changing potential.
    - Customer feedback: Mixed. Users acknowledge the potential benefits of the AI diagnostics but have reservations about accuracy, usability, and privacy concerns.
    - Product-market fit: Moderate. There's a clear need for what the startup offers, but adoption barriers and competition from established healthcare providers moderate its impact.
    - Sentiment: Neutral. Communications from the startup are hopeful but tempered by the realistic challenges of innovating in healthcare.
    - Innovation mention: Sometimes. The startup mentions innovation in its communications, but not to the extent that it differentiates significantly from competitors.
    - Technology emphasis: Mentioned. While AI is a focal point, the startup hasn't fully demonstrated how its technology surpasses existing solutions in practical, scalable ways.
    - Timing: Too Early. The startup's product enters a market that's not fully prepared for widespread AI adoption in healthcare, facing hurdles in user trust and regulatory approval.
    
    This example aims to provide a nuanced view that acknowledges Startup ABC's efforts and potential while recognizing the challenges it faces in the highly regulated and competitive healthcare industry.
    
    """
    categories = getResponse(System_Content, startup_inco, model)
    print("Categorical Separation:-------- ", categories)

    # Apply the safe parsing function to your data
    categories, errors = safe_parse_categories(categories)

    categories_extracted = extract_dict_values(categories,  'startup_analysis_responses')

    # Actually it turns out simply extracting could work
    categories_attempt = categories['startup_analysis_responses']

    # Encoding categorical features
    # Load the encoder
    encoder = joblib.load('models/trained_encoder_RF.joblib')

    # The adjusted category mappings with 'Mismatch' included
    category_mappings = {
        "industry_growth": ["No", "N/A", "Yes", "Mismatch"],
        "market_size": ["Small", "Medium", "Large", "N/A", "Mismatch"],
        "development_pace": ["Slower", "Same", "Faster", "N/A", "Mismatch"],
        "market_adaptability": ["Not Adaptable", "Somewhat Adaptable", "Very Adaptable", "N/A", "Mismatch"],
        "execution_capabilities": ["Poor", "Average", "Excellent", "N/A", "Mismatch"],
        "funding_amount": ["Below Average", "Average", "Above Average", "N/A", "Mismatch"],
        "valuation_change": ["Decreased", "Remained Stable", "Increased", "N/A", "Mismatch"],
        "investor_backing": ["Unknown", "Recognized", "Highly Regarded", "N/A", "Mismatch"],
        "reviews_testimonials": ["Negative", "Mixed", "Positive", "N/A", "Mismatch"],
        "product_market_fit": ["Weak", "Moderate", "Strong", "N/A", "Mismatch"],
        "sentiment_analysis": ["Negative", "Neutral", "Positive", "N/A", "Mismatch"],
        "innovation_mentions": ["Rarely", "Sometimes", "Often", "N/A", "Mismatch"],
        "cutting_edge_technology": ["No", "Mentioned", "Emphasized", "N/A", "Mismatch"],
        "timing": ["Too Early", "Just Right", "Too Late", "N/A", "Mismatch"]
    }

    # The order of features as used during training
    feature_order = list(category_mappings.keys())
    print(feature_order)

    # Call the function with our special encoder & trained model
    prediction = preprocess_and_predict(categories_attempt, category_mappings, encoder, model_random_forest, feature_order)

    if prediction[0] == 0:
        prediction = "Unsuccessful"
    if prediction[0] == 1:
        prediction = "Successful"

    print(f"Prediction: {prediction}")
    return prediction


def preprocess_and_predict(single_instance, category_mappings, encoder_special, model, feature_order):
    # Convert single instance dictionary into DataFrame
    single_instance_df = pd.DataFrame([single_instance])
    #single_instance_df = single_instance

    # Preprocess single_instance_df to match training feature names and order
    for column in feature_order:
        if column not in single_instance_df:
            single_instance_df[column] = "Mismatch"  # Add missing columns as "Mismatch"

    # Ensure DataFrame columns are in the same order as during training
    single_instance_df = single_instance_df[feature_order]

    # Replace categories not in mappings with "Mismatch"
    for column, categories in category_mappings.items():
        single_instance_df[column] = single_instance_df[column].apply(lambda x: x if x in categories else "Mismatch")

    print(encoder_special.categories_)
    # Encode the single instance using the trained OrdinalEncoder
    single_instance_encoded = encoder_special.transform(single_instance_df)

    # Use the trained model to predict
    prediction = model.predict(single_instance_encoded)

    return prediction


def safe_parse_categories(data):
    parsed_data = []
    error_indices = []

    # Check if data is a single string and convert it to a dictionary
    if isinstance(data, str):
        try:
            data = json.loads(data)  # Convert to dictionary
            data = pd.Series([data])  # Convert to Series for uniform handling
        except json.JSONDecodeError as e:
            print(f"Failed to parse string: {e}")
            return pd.DataFrame(parsed_data), [0]  # Return with error

    # Now handle as Series or DataFrame
    for index, row in data.items():
        try:
            parsed_row = json.loads(row) if isinstance(row, str) else row
            parsed_data.append(parsed_row)
        except json.JSONDecodeError:
            print(f"Failed to parse row at index {index}")
            error_indices.append(index)
            parsed_data.append({})

    return pd.DataFrame(parsed_data), error_indices


def extract_dict_values(df, column_name):
    keys = set()  # A set to hold all keys
    # First, find all keys used in the dictionaries
    for index, row in df.iterrows():
        keys.update(row[column_name].keys())
    keys = list(keys)  # Convert set to list

    # Now, for each key found, create a new column in the DataFrame
    for key in keys:
        df[f"{column_name}_{key}"] = df[column_name].apply(lambda d: d.get(key, None))

    return df.drop(column_name, axis=1)  # Drop the original dictionary column


def getEmbeddings(input_text):
    client = OpenAI(api_key=OPENAI_API_KEY, )
    response = client.embeddings.create(
        input=input_text,
        model="text-embedding-3-large",
        dimensions = 100
    )
    return(response.data[0].embedding)

# Function to calculate cosine similarity
def calculate_cosine_similarity(list1, list2):
    # Ensure inputs are numpy arrays
    arr1 = np.array(list1).reshape(1, -1)
    arr2 = np.array(list2).reshape(1, -1)
    # Calculate and return cosine similarity
    return cosine_similarity(arr1, arr2)[0][0]

def getFit(founder_info, startup_info):
    founder_embeddings = getEmbeddings(founder_info)
    startup_embeddings = getEmbeddings(startup_info)
    cosine_similarity = calculate_cosine_similarity(founder_embeddings, startup_embeddings)
    print(f"Similarity: {cosine_similarity}")

    # Assuming new_founder_info, new_startup_info_long, and new_cosine_similarity are your new data inputs
    X_new_embeddings = np.array(founder_embeddings).reshape(1, -1)
    X_new_embeddings_2 = np.array(startup_embeddings).reshape(1, -1)
    X_new_cosine = np.array([[cosine_similarity]])  # Reshape to (1, 1)

    print(f"Type of X_new_embeddings: {type(X_new_embeddings)}, shape: {X_new_embeddings.shape}")
    print(f"Type of X_new_embeddings_2: {type(X_new_embeddings_2)}, shape: {X_new_embeddings_2.shape}")
    print(f"Type of X_new_cosine: {type(X_new_cosine)}, shape: {X_new_cosine.shape}")

    X_new = np.concatenate([X_new_embeddings, X_new_embeddings_2, X_new_cosine], axis=1)

    # Normalize features
    #scaler = StandardScaler()
    #X_new_scaled = scaler.fit_transform(X_new)
    print(f"Input:{X_new}")
    prediction = model_neural_network.predict(X_new)
    return prediction

def LevelSegmentation(userContent, model):
    System_Content = f""""
    You are an analyst. Your task is to output one of the options: [‘L1’, ‘L2’, ‘L3’, ‘L4’, ‘L5’]. Do not output anything else. 

    You think step by step, and categorise according to the criteria below. Essentially, L5 is most successful and L1 is least successful. 

    * Level 5 (L5): Entrepreneur who has built a $100M+ ARR business. Similar definitions extend to a a founder who IPO'ed a company or took a company to public or a company that got sold over $500M. 
    * Level 4 (L4): Entrepreneur who has had a small to medium-size exit or has worked as an executive at a notable technology company
    * Level 3 (L3): 10-15 years of technical and management experience (e.g. working at big tech and unicorn startups or having a PhD)
    * Level 2 (L2): Entrepreneurs with a few years of experience or accelerator graduates
    * Level 1 (L1): Entrepreneurs that are disconnected from tech circles or that have negligible experience but with large potential

    Some examples are provided here:
    L5: Dan Siroker. He founded Optimizely as a CEO, which raised $251M and surpassed $100M ARR. Then, he sold this company for over $1B (over unicorn valuation). This is the ultimate outlier success. He is a repeat entrepreneur. 

    L4: Harry Glaser. He founded Priscope Data as a CEO, which raised $34M. Then he sold his company for close $100M. This is highly successful, but not a billion dollar outcome. He is a repeat entrepreneur. 

    L4: Sandeep Menon. He was a vice president at Google. VPs at Google manages large teams and own P&L. Google is a top-tier tech company and being a VP is an incredibly difficult position to get to. Though he is not a repeat entrepreneur, Sandeep is a highly successful executive. 

    L3: Alhussein Fawzi. He has a Phd in machine learning from EPFL, which is one of the top universities in the world. He worked as a researcher at DeepMind, which is one of the top-tier AI research companies in the world. He is a first-time founder with a strong expertise and pedigree as a researcher. 

    L3: Israel Shalom. He is a graduate of top universities such as Technion and Berkeley. He worked as a software engineer, product lead and group product manager at Google and Dropbox, which are top-tier employers. Working as a product manager is highly reputable in these companies, and being a product lead indicate the career growth. He is a first-time founder with 10-15 years of industry experience with a strong pedigree. 

    L2: Marcus Lowe. He is a graduate of MIT, one of the top universities. He worked as a product manager at Google. Graduating from MIT and working as a product manager at Google are hard things to achieve. He is a first-time entrepreneur with 4-5 years of experience with a strong pedigree. 

    L1: Anyone that does not fit into L2, L3, L4, L5 are in this category. They may be recent graduates with no experience, they may be university dropouts or they may be first-time entrepreneurs with less reputable pedigrees such as working at unknown companies or less successful companies. 

    """
    response = getResponse(System_Content, userContent, model)
    print("Segmenting:-------- ", response)
    return response

# Testing Area
test_startup = "WeLight aims to revolutionise China's $2.5 billion college application consulting market by increasing access for over a million Chinese students aspiring to study abroad. As an AI-powered platform, WeLight automates program selection, preparation guidance, and essay review using Large Language Models (LLM), the ANNOY Model, and an extensive database. Additionally, it facilitates mentor-mentee connections for skill development in interviews, English proficiency, and essay writing."

#report = externalMarketKnowledge("WeLight aims to revolutionise China's $2.5 billion college application consulting market by increasing access for over a million Chinese students aspiring to study abroad. As an AI-powered platform, WeLight automates program selection, preparation guidance, and essay review using Large Language Models (LLM), the ANNOY Model, and an extensive database. Additionally, it facilitates mentor-mentee connections for skill development in interviews, English proficiency, and essay writing.")

#SYS = "Explain to me in detail about the market: Chinese Education Consulting Market, Growth, Trend, Size, Revenue. Generate a report about the growth & size of the market. Also, give insights on the timing of entering the market now & also market sentiment. Make your response structured and in detail."
#print(getResponse(SYS,"",'gpt-4'))

founder_info = "Xisen Wang is the founder. He is from Oxford University, studying engineering science and having experiences in AI Research and founding an education NGO. He is 2nd year undergraduate right now, but has extensive networks and passion."

print(mainPipeline(test_startup, founder_info,"advanced", "gpt-4" ))

#print(getFit("?????????????","A startup of Generative AI. "))

# Combine the information together
#startup_info = "Startup Description:" + test_startup + "Founder Information: " + founder_info
#print(startup_info)

#sideVCScout(startup_info,'gpt-4')
