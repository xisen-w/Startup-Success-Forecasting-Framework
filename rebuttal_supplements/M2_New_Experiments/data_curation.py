'''This file contains the code for data curation for the new experiments.'''

'''Essentially, we can play around with the proportion of unsuccessful from the successful ones.'''

import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import time

def load_and_curate_data():
    print("🔄 STARTING DATA CURATION FOR M2 NEW EXPERIMENTS")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to the root 'data' directory, assuming it's two levels up from the script
    base_data_path_relative = os.path.join(script_dir, '..', '..', 'data') 
    base_data_path_absolute = os.path.normpath(base_data_path_relative) # Normalize the path

    successful_csv_path = os.path.join(base_data_path_absolute, 'Merged_Successful_V2.csv')
    unsuccessful_csv_path = os.path.join(base_data_path_absolute, 'Merged_Unsuccessful_V2.csv')

    print(f"Script directory: {script_dir}")
    print(f"Normalized base data directory: {base_data_path_absolute}")
    print(f"Attempting to load successful data from: {successful_csv_path}")
    print(f"Attempting to load unsuccessful data from: {unsuccessful_csv_path}")

    # First load the whole dataset (print the size of it for both successful and unsuccessful)
    print("\n📊 Loading datasets...")
    
    try:
        print(f"Reading successful_df: {successful_csv_path}...")
        start_time = time.time()
        successful_df = pd.read_csv(successful_csv_path, low_memory=False)
        end_time = time.time()
        print(f"✅ Successful companies loaded: {len(successful_df):,} records (took {end_time - start_time:.2f} seconds)")
    except FileNotFoundError:
        print(f"❌ ERROR: Successful data file not found at {successful_csv_path}")
        return
    except pd.errors.ParserError as e:
        print(f"❌ ERROR: Could not parse successful data file. Error: {e}")
        return
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading successful data: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print(f"Reading unsuccessful_df: {unsuccessful_csv_path}...")
        start_time = time.time()
        unsuccessful_df = pd.read_csv(unsuccessful_csv_path, low_memory=False)
        end_time = time.time()
        print(f"❌ Unsuccessful companies loaded: {len(unsuccessful_df):,} records (took {end_time - start_time:.2f} seconds)")
    except FileNotFoundError:
        print(f"❌ ERROR: Unsuccessful data file not found at {unsuccessful_csv_path}")
        return
    except pd.errors.ParserError as e:
        print(f"❌ ERROR: Could not parse unsuccessful data file. Error: {e}")
        return
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading unsuccessful data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"📈 Total original dataset size: {len(successful_df) + len(unsuccessful_df):,} companies")
    
    # Create output directory for curated files relative to the script's location
    output_dir = os.path.join(script_dir, 'data') 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Curated data will be saved to: {output_dir}")
    
    # Then from 0.1 to 0.2, 0.3, 0.4, 0.5. this is the ratio of successful from the whole new curated dataset.
    success_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # We want the size to be 1000 in total for evaluation. Everything is randomly sampled.
    total_size = 1000
    
    print(f"\n🎯 Creating curated datasets with {total_size} samples each")
    print("=" * 60)
    
    for ratio in success_ratios:
        print(f"\n📋 Creating dataset with {ratio:.1%} success rate...")
        
        # Calculate number of successful and unsuccessful samples needed
        num_successful = int(total_size * ratio)
        num_unsuccessful = total_size - num_successful
        
        print(f"   🏆 Successful samples needed: {num_successful}")
        print(f"   💔 Unsuccessful samples needed: {num_unsuccessful}")
        
        # Randomly sample from each dataset
        successful_sample = successful_df.sample(n=min(num_successful, len(successful_df)), 
                                               random_state=42).copy()
        unsuccessful_sample = unsuccessful_df.sample(n=min(num_unsuccessful, len(unsuccessful_df)), 
                                                   random_state=42).copy()
        
        # Add labels
        successful_sample['label'] = 1
        unsuccessful_sample['label'] = 0
        
        # Combine and shuffle
        combined_df = pd.concat([successful_sample, unsuccessful_sample], ignore_index=True)
        combined_df = shuffle(combined_df, random_state=42).reset_index(drop=True)
        
        # Print statistical summary for each of the new curated dataset
        print(f"\n📊 Statistical Summary for {ratio:.1%} success rate dataset:")
        print(f"   📏 Total size: {len(combined_df)}")
        print(f"   🏆 Successful: {len(successful_sample)} ({len(successful_sample)/len(combined_df):.1%})")
        print(f"   💔 Unsuccessful: {len(unsuccessful_sample)} ({len(unsuccessful_sample)/len(combined_df):.1%})")
        
        # Check if we have key columns for analysis
        if 'category_list' in combined_df.columns:
            print(f"   🏭 Industries covered: {combined_df['category_list'].nunique()} unique categories")
        if 'country_code' in combined_df.columns:
            print(f"   🌍 Countries covered: {combined_df['country_code'].nunique()} countries")
        if 'city' in combined_df.columns:
            print(f"   🏙️ Cities covered: {combined_df['city'].nunique()} cities")
        
        # Store the curated dataset under rebuttal_supplements/M2_New_Experiments/data
        filename = f"curated_dataset_success_{int(ratio*100)}pct.csv"
        filepath = os.path.join(output_dir, filename)
        combined_df.to_csv(filepath, index=False)
        print(f"   💾 Saved to: {filepath}")
        
        # Additional statistics
        print(f"   📈 Success rate verification: {combined_df['label'].mean():.1%}")
        
    print(f"\n✅ Data curation completed!")
    print(f"📁 All curated datasets saved in: {output_dir}")
    print("\n📋 Summary of created datasets:")
    for ratio in success_ratios:
        filename = f"curated_dataset_success_{int(ratio*100)}pct.csv"
        print(f"   • {filename} - {ratio:.1%} success rate, {total_size} samples")

if __name__ == "__main__":
    load_and_curate_data()




