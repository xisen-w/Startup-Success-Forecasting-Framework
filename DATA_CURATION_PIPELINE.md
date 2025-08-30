# Data Curation Pipeline for Startup Success Forecasting

This document outlines a conceptual pipeline for curating data suitable for a startup success forecasting framework like the SSFF. The goal is to gather comprehensive, publicly available information that mirrors the data points used by our system, as defined in the `StartupInfo` Pydantic schema within `agents/vc_scout_agent.py`.

While the specific datasets used in our original research contain proprietary information, this pipeline provides a methodology for researchers to construct analogous datasets.

## I. Objective

To collect and structure publicly available data on startups, focusing on:
1.  **Company Profile:** Basic information, description, industry.
2.  **Market Context:** Market size, growth, competition, trends.
3.  **Product Details:** Product description, technology, USP, scalability.
4.  **Founder Information:** Backgrounds, track records, relevant experience.
5.  **Traction & External Signals (Optional but Recommended):** Web traffic, social media, funding, news sentiment.

## II. Data Sources (Examples)

Researchers can leverage a variety of public sources:

*   **Startup Aggregators & Databases:**
    *   Crunchbase (offers free search and some data, paid API for more extensive access)
    *   Dealroom
    *   Tracxn
    *   AngelList
    *   Local/regional startup directories
*   **Company Websites:** "About Us," "Product," "Team," "News/Press" sections.
*   **Professional Networks:**
    *   LinkedIn (for founder backgrounds, team experience, company updates)
*   **News Articles & Press Releases:** Search engines (Google, Bing), industry-specific news sites.
*   **Product Review Sites:** G2, Capterra, TrustRadius (for user feedback on products).
*   **Financial Data Platforms (often require subscription):** PitchBook, Preqin (for detailed funding, M&A).
*   **SEO & Web Analytics Tools (some offer limited free data):** SEMrush, Ahrefs, SimilarWeb (for web traffic estimates).

## III. Data Collection & Structuring Workflow

This is a multi-step process, likely requiring a combination of automated scraping (where permissible and ethical) and manual research.

### Step 1: Startup Identification & Initial Split

1.  **Define Scope:** Determine criteria for startups to include (e.g., industry, stage, geography, timeframe of founding).
2.  **Initial List Generation:**
    *   Use startup aggregators (e.g., Crunchbase search filters) to generate an initial list of companies.
    *   Categorize them broadly (e.g., by potential success/failure if historical data is available and you are trying to replicate a labeled dataset, or simply as a cohort for prospective analysis). *Self-correction: For open-sourcing a general pipeline, the initial split might be more about identifying a target cohort rather than pre-labeling success/failure, as that's what the forecasting model aims to predict.*

### Step 2: Company & Market Data Collection

For each identified startup:

1.  **Company Profile:**
    *   **`name`**: Official name of the startup.
    *   **`description`**: Brief overview from their website, Crunchbase, etc.
2.  **Market Information (from news, reports, company statements):**
    *   **`market_size`**: Estimated target market size.
    *   **`growth_rate`**: Market growth rate (e.g., CAGR).
    *   **`competition`**: Key competitors.
    *   **`market_trends`**: Current relevant market trends.
    *   **`go_to_market_strategy`**: If available from company descriptions or interviews.

### Step 3: Product Data Collection

1.  **Product Details (from company website, product pages, reviews):**
    *   **`product_details`**: In-depth description of the product/service, key features.
    *   **`technology_stack`**: Mentioned technologies (can be hard to find consistently).
    *   **`scalability`**: Information on how the product is designed to scale.
    *   **`user_feedback`**: Summaries from review sites or public comments.
    *   **`product_fit`** (Unique Selling Proposition / Value Proposition): What makes the product unique or fit the market.

### Step 4: Founder Data Collection

This is often the most manual part.

1.  **Identify Founders:** From company website, Crunchbase, LinkedIn.
2.  **Gather Backgrounds (primarily from LinkedIn, founder bios):**
    *   **`founder_backgrounds`**: Summarize education, key previous roles, years of experience in relevant fields.
    *   **`track_records`**: Notable achievements in previous roles, previous startup experience (successes or failures).
    *   **`leadership_skills`**: Harder to quantify; look for mentions of team size led, significant projects managed.
    *   **`vision_alignment`**: Inferred from founder interviews, company mission statements.
    *   **`team_dynamics`**: Very difficult to ascertain from public data; likely to be `null` or based on high-level observations if multiple founders are profiled.

### Step 5: Optional Traction & External Signals

1.  **`web_traffic_growth`**: Use tools like SimilarWeb (if accessible) for trends.
2.  **`social_media_presence`**: Follower counts, engagement (requires manual checking or specialized tools).
3.  **`investment_rounds`**: From Crunchbase, news articles.
4.  **`regulatory_approvals`, `patents`**: If applicable to the industry (e.g., biotech, deep tech), search patent databases and news.

### Step 6: Data Structuring & Formatting

1.  **Consolidate Information:** For each startup, compile all collected data.
2.  **Schema Alignment:** Structure the data according to a defined schema, ideally matching the `StartupInfo` Pydantic model from `agents/vc_scout_agent.py`. This means each field in `StartupInfo` would be a column or key.
3.  **Input String Creation:** The SSFF framework's `analyze_startup` functions expect a single input string (`startup_info_str`). This string should be a coherent paragraph or a structured text that combines the most salient collected information. For example, in `run_experimentation_ssff_regular.py`, the 'paragraph' (from founder info) and 'long_description' (from company info) are concatenated. A more systematic approach would be to craft a template:

    ```text
    Company: [Name]
    Description: [Startup Description]
    Market: Market Size is [Market Size], Growth Rate is [Growth Rate]. Key competitors include [Competition]. Current trends are [Market Trends].
    Product: The product, [Product Details], utilizes [Technology Stack]. Its unique selling point is [Product Fit/USP].
    Founders: The founding team has backgrounds in [Founder Backgrounds summary] with track records including [Track Records summary].
    (Optional: Add other available fields like Investment Rounds, etc.)
    ```
4.  **Output Format:** Store the structured data (e.g., as a CSV or JSON file where each row/object is a startup) and/or the derived `startup_info_str` for each.

## IV. Ethical Considerations & Limitations

*   **Public Data Only:** This pipeline relies on publicly available information. Respect `robots.txt` for websites and terms of service for platforms.
*   **Data Accuracy & Bias:** Public data can be outdated, incomplete, or biased (e.g., overly positive company descriptions). Acknowledge these limitations.
*   **Manual Effort:** Collecting comprehensive data, especially for founders, can be time-intensive.
*   **Consistency:** Ensuring consistent data collection across many startups requires clear guidelines and potentially multiple researchers cross-validating.

## V. Tools & Technologies (Suggestions)

*   **Programming Language:** Python (for scripting, web scraping, data processing).
*   **Web Scraping Libraries (use responsibly):** `BeautifulSoup`, `Scrapy`, `Requests`.
*   **Browser Automation (for dynamic content):** `Selenium`, `Playwright`.
*   **Data Handling:** `Pandas` for data manipulation and storage.
*   **NLP (Optional, for pre-processing text):** Libraries like `spaCy` or `NLTK` if further text processing is needed before feeding to LLMs.

This pipeline provides a starting point. Researchers will need to adapt it based on their specific research questions, resources, and the types of startups they are analyzing. 