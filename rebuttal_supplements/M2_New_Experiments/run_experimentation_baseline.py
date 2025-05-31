# This file is bult on the previous fle for a detailed experimentation.ExceptionGroup
    
#Essentially, we do this for each experiment. 

# 4o-mini, 4o, o1-mini, o3-mini. (four groups of models)

# and then for each model, we are basically calling ssff framework but here's the thing

# we also have baseline methods (intotal we compare 6 methods): method 1 pure vannila GPT. method 2: COT prompting. 3: FoundersGPT 4: RAISE Paper. 5.Without ML 6.With ML.

# But for this draft. we focus on the pure vanilla GPT

# We also like the output to be like jsonl and also use tqdm to show the progress. We want after each round, we save another line. 

# The data sets are like: rebuttal_supplements/M2_New_Experiments/data/curated_dataset_success_10pct.csv to 50pct now. try to make it that we can use the command to choose which dataset to use. the 10pct is the default.

# also enable in the command line to choose the model (while multiple models are supported) and then also support :n (like how many lines of input do we processs)

import os
import sys
import logging
import argparse
import pandas as pd
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the project root directory to the Python path
# Assuming this script is in rebuttal_supplements/M2_New_Experiments/
# and the 'agents' directory is at the project root level,
# which is two levels up from this script's current directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Ensure the agents module can be found
# If agents.base_agent is still not found, you might need to adjust the path
# or ensure __init__.py files are present in relevant directories.
try:
    from agents.base_agent import BaseAgent
except ImportError:
    print("Error: Could not import BaseAgent. Please check the sys.path and project structure.")
    print(f"Current sys.path includes: {project_root}")
    sys.exit(1)


class BaselineAnalysis(BaseModel):
    total_analysis: str = Field(..., description="Detailed analysis of the startup")
    score: float = Field(..., description="Overall score between 1 and 10")
    recommendation: str = Field(..., description="Recommendation: 'Successful' or 'Unsuccessful'")

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def analyze_startup(self, startup_info: str) -> dict:
        """Simple baseline analysis using only ChatGPT"""
        self.logger.debug(f"Starting baseline analysis for: {startup_info[:100]}...") # Log snippet
        
        prompt = """
        You are an experienced venture capitalist analyzing a startup. Based on the provided information, 
        give a comprehensive analysis and predict if the startup will be successful or not.
        
        Your analysis should include:
        1. Market analysis
        2. Product/technology evaluation
        3. Founder/team assessment
        4. Overall score (1-10)
        5. Investment recommendation (must be 'Successful' or 'Unsuccessful')
        """
        
        try:
            # Assuming startup_info is the raw text description
            response = self.get_json_response(BaselineAnalysis, prompt, startup_info)
            return response.dict()
            
        except Exception as e:
            self.logger.error(f"Error in baseline analysis for startup info '{startup_info[:100]}...': {str(e)}")
            return {"error": str(e), "startup_info_processed": startup_info}

def main():
    parser = argparse.ArgumentParser(description="Run baseline startup analysis experiments.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/curated_dataset_success_10pct.csv",
        help="Path to the input CSV dataset, relative to the script's directory's 'data' subdirectory."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"], # As per your comments
        help="LLM model to use for analysis."
    )
    parser.add_argument(
        "--num_rows", 
        type=int, 
        default=None,
        help="Number of rows to process from the dataset. Processes all if not specified."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path to the output JSONL file. Defaults to a generated name."
    )
    parser.add_argument(
        "--info_column",
        type=str,
        default="description", # Assuming 'description' is the column with startup text
        help="Name of the column in the CSV that contains the startup information text."
    )

    args = parser.parse_args()

    # Construct full dataset path
    # The script is in M2_New_Experiments, data is in M2_New_Experiments/data
    script_dir = os.path.dirname(__file__)
    full_dataset_path = os.path.join(script_dir, args.dataset_path)


    if args.output_file is None:
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        args.output_file = os.path.join(script_dir, f"results_baseline_{args.model_name}_{dataset_name}.jsonl")

    logging.info(f"Starting experiment with an LLM model: {args.model_name}")
    logging.info(f"Processing dataset: {full_dataset_path}")
    if args.num_rows:
        logging.info(f"Number of rows to process: {args.num_rows}")
    logging.info(f"Output will be saved to: {args.output_file}")
    logging.info(f"Startup info will be read from column: '{args.info_column}'")

    try:
        df = pd.read_csv(full_dataset_path)
        logging.info(f"Successfully loaded dataset. Total rows: {len(df)}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {full_dataset_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.info_column not in df.columns:
        logging.error(f"Specified info_column '{args.info_column}' not found in the dataset.")
        logging.error(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    if args.num_rows:
        df_to_process = df.head(args.num_rows)
    else:
        df_to_process = df

    framework = BaselineFramework(model=args.model_name)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w') as outfile:
        for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing startups"):
            startup_info_text = row[args.info_column]
            if pd.isna(startup_info_text):
                logging.warning(f"Skipping row {index} due to missing startup info in column '{args.info_column}'.")
                result = {"error": "Missing startup info", "original_index": index}
            else:
                result = framework.analyze_startup(str(startup_info_text))
                result["original_index"] = index # Keep track of original row

            outfile.write(json.dumps(result) + '\\n')
    
    logging.info(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()