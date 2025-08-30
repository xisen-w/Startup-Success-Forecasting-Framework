import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import json
import ast # For parsing string representations of dicts
from datetime import datetime
import os # Added for path operations and directory creation

# Helper function to parse dates like "dYYYY-MM-XX" or "YYYY-MM-DD" or just "YYYY"
def parse_employment_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.replace('d', '') # Remove 'd' prefix if present
    parts = date_str.split('-')
    try:
        if len(parts) == 1 and parts[0].strip().lower() != 'xx': # YYYY
            return datetime(int(parts[0]), 1, 1) # Assume Jan 1st if only year
        elif len(parts) >= 2 and parts[0].strip().lower() != 'xx' and parts[1].strip().lower() != 'xx': # YYYY-MM
            year = int(parts[0])
            month = int(parts[1])
            day = 1
            if len(parts) == 3 and parts[2].strip().lower() != 'xx': # YYYY-MM-DD
                day = int(parts[2])
            return datetime(year, month, day)
    except ValueError:
        return None
    return None

def calculate_experience_years(employments):
    total_experience_days = 0
    if not employments or not isinstance(employments, list):
        return 0

    for emp in employments:
        if not isinstance(emp, dict):
            continue
        
        start_date_str = emp.get('from')
        if isinstance(start_date_str, dict): # Handle cases where 'from' is a dict like {'str': 'd2012-07-XX', ...}
            start_date_str = start_date_str.get('str')
            
        end_date_str = emp.get('to')
        if isinstance(end_date_str, dict):
            end_date_str = end_date_str.get('str')

        start_date = parse_employment_date(start_date_str)
        
        if emp.get('isCurrent', False) and start_date: # if current, count until today
            end_date = datetime.now()
        else:
            end_date = parse_employment_date(end_date_str)

        if start_date and end_date and end_date > start_date:
            total_experience_days += (end_date - start_date).days
            
    return total_experience_days / 365.25 # Average days in a year

def has_prior_startup_experience(employments, current_org_name=None):
    if not employments or not isinstance(employments, list):
        return False
    
    startup_keywords = ['founder', 'co-founder', 'cto', 'ceo', 'chief', 'president'] # Simplified
    # Consider an employment as a startup if title contains keywords, or if employer is different from current org
    # and role indicates significant leadership/founding.

    for emp in employments:
        if not isinstance(emp, dict):
            continue

        title = str(emp.get('title', '')).lower()
        employer_info = emp.get('employer')
        employer_name = None
        if isinstance(employer_info, dict):
            employer_name = str(employer_info.get('name', '')).lower()
        elif isinstance(employer_info, str):
            employer_name = employer_info.lower()

        # Check if the title contains startup-like keywords
        if any(keyword in title for keyword in startup_keywords):
            # If a current_org_name is provided, ensure this role is not at the current company
            # to count as *prior* startup experience.
            # This simplistic check might need refinement if founders had multiple roles at current company.
            if current_org_name and employer_name and current_org_name.lower() == employer_name:
                if not emp.get('isCurrent', False): # If it is a past role at the same company, could still be prior internal venture.
                    return True # Or decide how to handle this.
            else:
                 # If it's a different company OR if no current_org_name is specified, count it.
                return True
    return False

def extract_education_levels(educations):
    levels = []
    if not educations or not isinstance(educations, list):
        return levels

    for edu in educations:
        if not isinstance(edu, dict):
            continue
        degree_info = edu.get('degree')
        degree_name = None
        if isinstance(degree_info, dict):
            degree_name = degree_info.get('name') or degree_info.get('surfaceForm')
        elif isinstance(degree_info, str):
            degree_name = degree_info
        
        if degree_name and isinstance(degree_name, str):
            # Normalize common degree names
            dn_lower = degree_name.lower()
            if 'phd' in dn_lower or 'doctorate' in dn_lower:
                levels.append('PhD/Doctorate')
            elif "master's" in dn_lower or 'master' in dn_lower or 'mba' in dn_lower:
                levels.append("Master's/MBA")
            elif "bachelor's" in dn_lower or 'bachelor' in dn_lower or 'bs' in dn_lower or 'ba' in dn_lower:
                levels.append("Bachelor's")
            elif 'associate' in dn_lower:
                levels.append("Associate's")
            elif degree_name.strip(): # Add if not empty and not categorized
                levels.append(degree_name.strip()) # Keep original if not easily categorized
    return levels

def summarize_founder_experience(df):
    print("\n👤 FOUNDER EXPERIENCE SUMMARY:")
    
    all_founder_experience_years = []
    founders_with_prior_startup_exp = 0
    total_founders_processed_for_prior_exp = 0
    all_education_levels = []
    processed_orgs_for_founder_summary = 0

    for index, row in df.iterrows():
        founder_data_list = []
        # Try parsing json_string first
        if pd.notna(row['json_string']) and isinstance(row['json_string'], str):
            try:
                parsed_json = json.loads(row['json_string'])
                if isinstance(parsed_json.get('data'), list):
                    founder_data_list = parsed_json['data']
            except json.JSONDecodeError:
                pass # Will try structured_info next
        
        # If json_string fails or is not present, try structured_info
        if not founder_data_list and pd.notna(row['structured_info']) and isinstance(row['structured_info'], str):
            try:
                # Ensure it's a string that looks like a dict/list before ast.literal_eval
                eval_data = ast.literal_eval(row['structured_info'])
                if isinstance(eval_data, list): # If structured_info is a list of founders directly
                    founder_data_list = eval_data
                elif isinstance(eval_data, dict): # If structured_info is a single founder dict
                    founder_data_list = [eval_data]
            except (ValueError, SyntaxError):
                pass # Cannot parse this row's structured_info
        
        if not founder_data_list:
            continue # No founder data extracted for this company

        processed_orgs_for_founder_summary += 1
        company_had_founder_with_prior_exp = False
        for founder in founder_data_list:
            if not isinstance(founder, dict):
                continue

            employments = founder.get('employments')
            educations = founder.get('educations')
            current_org_name = row.get('org_name') 

            exp_years = calculate_experience_years(employments)
            if exp_years > 0: 
                all_founder_experience_years.append(exp_years)

            if employments is not None: 
                total_founders_processed_for_prior_exp +=1 
                if has_prior_startup_experience(employments, current_org_name):
                    company_had_founder_with_prior_exp = True
            
            edu_levels = extract_education_levels(educations)
            all_education_levels.extend(edu_levels)
        
        if company_had_founder_with_prior_exp:
            founders_with_prior_startup_exp +=1 

    print(f"\n  (Processed {processed_orgs_for_founder_summary} companies for founder details)")

    # --- Average years of experience (overall for all founders found) ---
    avg_exp_summary = None
    print("\n  ⏳ Average Years of Experience (across all founders with data):")
    if all_founder_experience_years:
        avg_experience = np.mean(all_founder_experience_years)
        print(f"    Overall average years of professional experience: {avg_experience:.2f} years")
        print(f"    (Based on {len(all_founder_experience_years)} founders with calculable experience from {processed_orgs_for_founder_summary} companies)")
        avg_exp_summary = pd.DataFrame([{
            'Metric': 'Overall Average Years of Professional Experience',
            'Value': f"{avg_experience:.2f}",
            'Basis': f"{len(all_founder_experience_years)} founders from {processed_orgs_for_founder_summary} companies"
        }])
    else:
        print("    Could not calculate average years of experience. No valid employment data found or parsed.")
        avg_exp_summary = pd.DataFrame([{'Metric': 'Overall Average Years of Professional Experience', 'Value': 'N/A', 'Basis': 'No data'}])

    # --- Percentage of founders with prior startup experience ---
    prior_exp_summary = None
    print("\n  🚀 Percentage of Companies with at least one Founder with Prior Startup Experience:")
    if processed_orgs_for_founder_summary > 0:
        percentage_prior_startup = (founders_with_prior_startup_exp / processed_orgs_for_founder_summary * 100)
        print(f"    Percentage of companies with prior founder startup experience: {percentage_prior_startup:.2f}% ({founders_with_prior_startup_exp} out of {processed_orgs_for_founder_summary} companies)")
        prior_exp_summary = pd.DataFrame([{
            'Metric': 'Percentage of Companies with Prior Founder Startup Experience',
            'Value': f"{percentage_prior_startup:.2f}%",
            'Basis': f"{founders_with_prior_startup_exp} out of {processed_orgs_for_founder_summary} companies"
        }])
    else:
        print("    Could not calculate percentage with prior startup experience. No founder employment data processed.")
        prior_exp_summary = pd.DataFrame([{'Metric': 'Percentage of Companies with Prior Founder Startup Experience', 'Value': 'N/A', 'Basis': 'No data'}])

    # --- Distribution of founders by education level ---
    education_summary_df = None
    print("\n  🎓 Distribution by Education Level (across all qualifications found):")
    if all_education_levels:
        education_counts = Counter(all_education_levels)
        total_degrees = len(all_education_levels)
        print(f"    Education Level Distribution (Total degrees/qualifications found: {total_degrees}):")
        
        edu_data = []
        for level, count in education_counts.most_common(): # Iterate through most_common for sorted output
            percentage = (count / total_degrees * 100) if total_degrees > 0 else 0
            print(f"      - {level}: {count} ({percentage:.2f}%)")
            edu_data.append({'Education Level': level, 'Count': count, 'Percentage': percentage})
        education_summary_df = pd.DataFrame(edu_data)
    else:
        print("    Could not determine education level distribution. No education data found or parsed.")
        education_summary_df = pd.DataFrame(columns=['Education Level', 'Count', 'Percentage'])
    
    return avg_exp_summary, prior_exp_summary, education_summary_df

def investigate_potential_founder_data_columns(df, num_samples=3):
    print("\n🕵️ INVESTIGATING POTENTIAL FOUNDER DATA COLUMNS 🕵️")
    potential_columns = ['json_string', 'structured_info', 'paragraph', 'integrated_info']
    
    for col_name in potential_columns:
        print(f"\n--- Samples from column: {col_name} ---")
        if col_name in df.columns:
            non_null_samples = df[col_name].dropna().head(num_samples)
            if not non_null_samples.empty:
                for i, sample in enumerate(non_null_samples):
                    print(f"Sample {i+1}:\n{sample}\n")
                if col_name in ['json_string', 'structured_info']:
                    print(f"Attempting to parse first non-null sample from '{col_name}' as JSON/dict...")
                    try:
                        first_sample = df[col_name].dropna().iloc[0]
                        parsed_data = None
                        if isinstance(first_sample, str):
                            if col_name == 'json_string':
                                parsed_data = json.loads(first_sample)
                                print(f"Successfully parsed '{col_name}' as JSON.")
                            elif col_name == 'structured_info':
                                parsed_data = ast.literal_eval(first_sample)
                                print(f"Successfully parsed '{col_name}' with ast.literal_eval.")
                            
                            if parsed_data:
                                if isinstance(parsed_data, dict):
                                    if 'data' in parsed_data and isinstance(parsed_data['data'], list) and parsed_data['data']:
                                        print(f"  Top-level keys of the dictionary: {list(parsed_data.keys())}")
                                        print(f"  Keys of the first item in 'data' list: {list(parsed_data['data'][0].keys()) if isinstance(parsed_data['data'][0], dict) else 'Not a dict'}")
                                    else: 
                                        print(f"  Keys found in parsed dict: {list(parsed_data.keys())}")
                                elif isinstance(parsed_data, list) and parsed_data:
                                    print(f"  Parsed as a list with {len(parsed_data)} items. Keys of first item (if dict):")
                                    if isinstance(parsed_data[0], dict):
                                        for key in parsed_data[0].keys():
                                            print(f"    - {key}")
                                print("\n")
                        else:
                            print(f"First sample from '{col_name}' is not a string, cannot parse.\n")
                    except Exception as e:
                        print(f"Could not parse first sample from '{col_name}': {e}\n")
            else:
                print(f"No non-null samples found in '{col_name}'.")
        else:
            print(f"Column '{col_name}' not found in the dataset.")
    print("="*50)

def summarize_sector_mix(df, top_n=20):
    print(f"\n🏭 TOP {top_n} INDUSTRIES/SECTORS (from 'category_list'):")
    all_categories = []
    for categories_str in df['category_list'].dropna():
        if pd.notna(categories_str) and categories_str != '':
            cats = [cat.strip() for cat in str(categories_str).split(',')]
            all_categories.extend(cats)
    if not all_categories:
        print("  No category data found or 'category_list' column is empty/missing.")
        return pd.DataFrame(columns=["Sector", "Total Companies", "Raw Mentions", "Successful", "Success Rate (%)"]) # Return empty DataFrame
    
    category_counter = Counter(all_categories)
    top_categories = category_counter.most_common(top_n)
    summary_data = []
    for category, count in top_categories:
        if category and category.strip():
            pattern = re.escape(category)
            category_companies_df = df[df['category_list'].str.contains(pattern, na=False, case=False, regex=True)]
            if not category_companies_df.empty:
                successful_count = len(category_companies_df[category_companies_df['outcome'] == 'Successful'])
                total_in_category = len(category_companies_df)
                success_rate = (successful_count / total_in_category * 100) if total_in_category > 0 else 0
                summary_data.append({
                    "Sector": category,
                    "Total Companies": total_in_category,
                    "Raw Mentions": count,
                    "Successful": successful_count,
                    "Success Rate (%)": success_rate
                })
                print(f"  - {category}: {total_in_category:,} companies ({success_rate:.1f}% success rate, {count:,} raw mentions)")
            else:
                print(f"  - {category}: 0 companies found with str.contains (Raw mentions: {count}). Check pattern or data.")
    return pd.DataFrame(summary_data)

def summarize_geography(df, top_n_countries=15, top_n_cities=15):
    print(f"\n🌍 TOP {top_n_countries} COUNTRIES (from 'country_code'):")
    country_summary_df = pd.DataFrame(columns=["Country Code", "Total Companies", "Successful", "Unsuccessful", "Success Rate (%)"])
    if 'country_code' not in df.columns:
        print("  'country_code' column not found.")
    else:
        country_analysis = df.groupby(['country_code', 'outcome']).size().unstack(fill_value=0)
        country_totals = country_analysis.sum(axis=1).sort_values(ascending=False).head(top_n_countries)
        country_summary_data = []
        for country_code, total_count in country_totals.items():
            if pd.notna(country_code):
                successful = country_analysis.loc[country_code, 'Successful'] if 'Successful' in country_analysis.columns and country_code in country_analysis.index else 0
                unsuccessful = country_analysis.loc[country_code, 'Unsuccessful'] if 'Unsuccessful' in country_analysis.columns and country_code in country_analysis.index else 0
                actual_total_for_rate = successful + unsuccessful 
                success_rate = (successful / actual_total_for_rate * 100) if actual_total_for_rate > 0 else 0
                country_summary_data.append({
                    "Country Code": country_code,
                    "Total Companies": actual_total_for_rate,
                    "Successful": successful,
                    "Unsuccessful": unsuccessful,
                    "Success Rate (%)": success_rate
                })
                print(f"  - {country_code}: {actual_total_for_rate:,} companies ({success_rate:.1f}% success rate)")
        if country_summary_data: # Ensure list is not empty before creating DataFrame
            country_summary_df = pd.DataFrame(country_summary_data)

    print(f"\n🏙️ TOP {top_n_cities} CITIES (from 'city' and 'country_code'):")
    city_summary_df = pd.DataFrame(columns=["City", "Country Code", "Total Companies", "Successful", "Success Rate (%)"])
    if 'city' not in df.columns or 'country_code' not in df.columns:
        print("  'city' or 'country_code' column not found.")
    else:
        city_analysis_counts = df.groupby(['city', 'country_code']).size().sort_values(ascending=False).head(top_n_cities)
        city_summary_data = []
        for (city, country_code), count in city_analysis_counts.items():
            if pd.notna(city) and pd.notna(country_code):
                city_country_df = df[(df['city'] == city) & (df['country_code'] == country_code)]
                successful_count = len(city_country_df[city_country_df['outcome'] == 'Successful'])
                total_in_city_country = len(city_country_df)
                success_rate = (successful_count / total_in_city_country * 100) if total_in_city_country > 0 else 0
                city_summary_data.append({
                    "City": city,
                    "Country Code": country_code,
                    "Total Companies": total_in_city_country,
                    "Successful": successful_count,
                    "Success Rate (%)": success_rate
                })
                print(f"  - {city}, {country_code}: {total_in_city_country:,} companies ({success_rate:.1f}% success rate)")
        if city_summary_data:
            city_summary_df = pd.DataFrame(city_summary_data)
    
    return country_summary_df, city_summary_df

def generate_descriptive_summary():
    print("🚀 Generating Descriptive Summary Table")
    print("="*50)
    
    output_dir = 'rebuttal_supplements/M1_Statistical_Summary/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"ℹ️ Summary CSV files will be saved to: {os.path.abspath(output_dir)}")

    path_successful = 'data/Merged_Successful_V2.csv'
    path_unsuccessful = 'data/Merged_Unsuccessful_V2.csv'
    try:
        print(f"🔄 Loading successful companies data from: {path_successful}")
        df_successful = pd.read_csv(path_successful)
        print(f"✅ Successfully loaded {len(df_successful)} successful companies.")
        print(f"🔄 Loading unsuccessful companies data from: {path_unsuccessful}")
        df_unsuccessful = pd.read_csv(path_unsuccessful)
        print(f"✅ Successfully loaded {len(df_unsuccessful)} unsuccessful companies.")
        df_successful['outcome'] = 'Successful'
        df_unsuccessful['outcome'] = 'Unsuccessful'
        print("🔗 Combining datasets...")
        df_all = pd.concat([df_successful, df_unsuccessful], ignore_index=True)
        print(f"📈 Combined dataset created with {len(df_all)} total companies.")
        print(f"   ({len(df_successful)} successful, {len(df_unsuccessful)} unsuccessful)")
        # print("\n📋 Available columns in the combined dataset:") # Optional: Keep for debugging
        # for i, col in enumerate(df_all.columns):
        #     print(f"  {i+1:2d}. {col}")
        
        # investigate_potential_founder_data_columns(df_all) 

        print("\n✨ Basic Class Ratios:")
        total_companies = len(df_all)
        successful_count = len(df_successful)
        unsuccessful_count = len(df_unsuccessful)
        class_ratios_data = []
        if total_companies > 0:
            success_rate = (successful_count / total_companies) * 100
            unsuccessful_rate = (unsuccessful_count / total_companies) * 100
            print(f"  Total Companies: {total_companies:,}")
            print(f"  Successful: {successful_count:,} ({success_rate:.2f}%)")
            print(f"  Unsuccessful: {unsuccessful_count:,} ({unsuccessful_rate:.2f}%)")
            class_ratios_data = [
                {'Metric': 'Total Companies', 'Count': total_companies, 'Percentage': 100.0},
                {'Metric': 'Successful Companies', 'Count': successful_count, 'Percentage': success_rate},
                {'Metric': 'Unsuccessful Companies', 'Count': unsuccessful_count, 'Percentage': unsuccessful_rate}
            ]
        else:
            print("  No company data found to calculate ratios.")
        class_ratios_df = pd.DataFrame(class_ratios_data)
        class_ratios_df.to_csv(os.path.join(output_dir, 'class_ratios_summary.csv'), index=False)
        print(f"💾 Class ratios summary saved to class_ratios_summary.csv")
        
        sector_summary_df = summarize_sector_mix(df_all)
        sector_summary_df.to_csv(os.path.join(output_dir, 'sector_mix_summary.csv'), index=False)
        print(f"💾 Sector mix summary saved to sector_mix_summary.csv")

        country_summary_df, city_summary_df = summarize_geography(df_all)
        country_summary_df.to_csv(os.path.join(output_dir, 'country_summary.csv'), index=False)
        print(f"💾 Country summary saved to country_summary.csv")
        city_summary_df.to_csv(os.path.join(output_dir, 'city_summary.csv'), index=False)
        print(f"💾 City summary saved to city_summary.csv")
        
        avg_exp_df, prior_exp_df, education_df = summarize_founder_experience(df_all)
        if avg_exp_df is not None: avg_exp_df.to_csv(os.path.join(output_dir, 'founder_average_experience.csv'), index=False)
        if prior_exp_df is not None: prior_exp_df.to_csv(os.path.join(output_dir, 'founder_prior_startup_experience.csv'), index=False)
        if education_df is not None: education_df.to_csv(os.path.join(output_dir, 'founder_education_distribution.csv'), index=False)
        print(f"💾 Founder experience summaries saved.")

        print("\n✅ Full descriptive summary generation and saving completed.")
        print("\nℹ️ Note: Plot generation is not yet implemented. If plots were generated, they would be saved in the same directory.")
        return df_all
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}")
        print("Please ensure the CSV files are correctly named and located at:")
        print(f"  - {path_successful}")
        print(f"  - {path_unsuccessful}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    combined_data = generate_descriptive_summary()
    if combined_data is not None:
        print("\nScript finished. Review the printed summaries and output CSV files.") 