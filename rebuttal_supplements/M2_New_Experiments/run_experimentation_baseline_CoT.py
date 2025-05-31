# This file is bult on the previous fle for a detailed experimentation.ExceptionGroup
    
#Essentially, we do this for each experiment. 

# 4o-mini, 4o, o1-mini, o3-mini. (four groups of models)

# and then for each model, we are basically calling ssff framework but here's the thing

# we also have baseline methods (intotal we compare 6 methods): method 1 pure vannila GPT. method 2: COT prompting. 3: FoundersGPT 4: RAISE Paper. 5.Without ML 6.With ML.

# Now revise upon this draft, which only has the ssff framework full version i think.

# We also like the output to be like jsonl and also use tqdm to show the progress. We want after each round, we save another line. 

import sys
import os
import json
import pandas as pd
from tqdm import tqdm
import logging
import traceback
import warnings
from datetime import datetime
from sklearn.exceptions import InconsistentVersionWarning
import csv

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ssff_framework import StartupFramework

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_dict_data(prefix, data_dict):
    """Helper function to dynamically extract all keys from a dictionary with a prefix"""
    if not isinstance(data_dict, dict):
        return {f"{prefix}": data_dict}
    
    return {f"{prefix}_{key}": value for key, value in data_dict.items()}

def run_natural_language_experiment():
    # Initialize framework
    framework = StartupFramework("gpt-4")
    
    # Load the experiment dataset
    input_path = os.path.join(project_root, 'data', 'Experiment_Dataset.csv') # Revise this to dynamically choose the exp dataset from our curated 5.
    if os.path.exists(input_path):
        startup_data = pd.read_csv(input_path)
    else:
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
    
    # Create results directory and CSV file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/results/natural_language_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = f"{results_dir}/complete_4o_results.csv"
    
    # Process each startup
    logger.info("Starting natural language startup analysis...")
    for idx, row in tqdm(startup_data.iterrows(), total=len(startup_data), desc="Processing startups"):
        try:
            # Get company name and success label
            company_name = row['company_name'] if 'company_name' in row else f"Unnamed Company {idx}"
            success_label = row['success']
            
            # Create startup info string
            startup_info_str = f"""
            Company: {company_name}
            {row['long_description']}
            Founder background: {row['paragraph']}
            """
            
            # Run natural language analysis
            result = framework.analyze_startup_natural(startup_info_str)
            
            # Store detailed analysis in separate files
            startup_name = f"startup_{idx}"
            
            # Store raw analysis results
            with open(f"{results_dir}/{startup_name}_raw.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            # Store natural language reports
            with open(f"{results_dir}/{startup_name}_reports.txt", 'w') as f:
                f.write("MARKET ANALYSIS:\n")
                f.write("=" * 40 + "\n")
                market_result = result.get('Market Analysis', {})
                f.write(market_result.get('analysis', 'No market analysis available'))
                
                f.write("\n\nPRODUCT ANALYSIS:\n")
                f.write("=" * 40 + "\n")
                product_result = result.get('Product Analysis', {})
                f.write(product_result.get('analysis', 'No product analysis available'))
                
                f.write("\n\nFOUNDER ANALYSIS:\n")
                f.write("=" * 40 + "\n")
                founder_result = result.get('Founder Analysis', {})
                f.write(founder_result.get('analysis', 'No founder analysis available'))
                
                f.write("\n\nFINAL INTEGRATED ANALYSIS:\n")
                f.write("=" * 40 + "\n")
                final_result = result.get('Final Analysis', {})
                f.write(str(final_result.get('analysis', 'No final analysis available')))
            
            # Create results dictionary for this startup
            results_dict = {
                'Company_Name': company_name,
                'Success_Label': success_label,
                # Input Data
                'Description': row['long_description'],
                'Founder Background': row['paragraph'],
            }
            
            # Store external knowledge reports and analyses
            if isinstance(result.get('Market Analysis'), dict):
                market_data = result['Market Analysis']
                results_dict.update({
                    'Market_Analysis': market_data.get('analysis', ''),
                    'Market_External_Report': market_data.get('external_report', '')
                })
            
            if isinstance(result.get('Product Analysis'), dict):
                product_data = result['Product Analysis']
                results_dict.update({
                    'Product_Analysis': product_data.get('analysis', ''),
                    'Product_External_Report': product_data.get('external_report', '')
                })
            
            # Dynamically add all keys from Founder Analysis
            if 'Founder Analysis' in result:
                results_dict.update(extract_dict_data('Founder', result['Founder Analysis']))
            
            # Dynamically add all keys from Final Analysis
            if 'Final Analysis' in result:
                results_dict.update(extract_dict_data('Final', result['Final Analysis']))
            
            # Add any other direct fields you want to keep
            results_dict.update({
                'Founder_Idea_Fit': result.get('Founder Idea Fit'),
                'Categorical_Prediction': result.get('Categorical Prediction'),
                'Quantitative_Decision': result.get('Quantitative Decision', {}).get('decision'),
                'Decision_Confidence': result.get('Quantitative Decision', {}).get('confidence'),
                'Decision_Rationale': result.get('Quantitative Decision', {}).get('rationale')
            })
            
            # Convert single result to DataFrame and append to CSV
            single_result_df = pd.DataFrame([results_dict])
            
            # Write to CSV with proper quoting
            if os.path.exists(results_csv_path):
                single_result_df.to_csv(results_csv_path, 
                                       mode='a', 
                                       header=False, 
                                       index=False,
                                       quoting=csv.QUOTE_ALL,  # Quote all fields
                                       escapechar='\\')  # Use backslash as escape character
            else:
                single_result_df.to_csv(results_csv_path, 
                                       mode='w', 
                                       header=True, 
                                       index=False,
                                       quoting=csv.QUOTE_ALL,
                                       escapechar='\\')
            
            logger.info(f"\nCompleted and saved analysis for startup {idx + 1}/{len(startup_data)}: {company_name}")
            
        except Exception as e:
            logger.error(f"Error processing startup {idx + 1}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info("\nNatural Language Analysis complete!")
    logger.info(f"Results saved to '{results_csv_path}'")

if __name__ == "__main__":
    run_natural_language_experiment()