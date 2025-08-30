import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import argparse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define scores to analyze and their paths within the JSON structure
# (main_key, sub_key_or_none_if_direct)
SCORES_TO_ANALYZE = {
    'market_viability': ('market_analysis_structured', 'viability_score'),
    'product_potential': ('product_analysis_structured', 'potential_score'),
    'product_innovation': ('product_analysis_structured', 'innovation_score'),
    'product_market_fit': ('product_analysis_structured', 'market_fit_score'),
    'founder_competency': ('founder_analysis_structured', 'competency_score'),
    'founder_idea_fit': ('founder_analysis_structured', 'idea_fit'),
    'pro_overall': ('final_analysis_pro_structured', 'overall_score'),
    'basic_overall': ('basic_analysis_structured', 'overall_score'),
    'founder_segmentation': ('founder_segmentation_value', None), # Direct key
    'top_founder_idea_fit': ('founder_idea_fit_score', None) # Direct key, often a duplicate of the one in founder_analysis_structured
}

SCORE_DISPLAY_NAMES = {
    'market_viability': 'Market Viability',
    'product_potential': 'Product Potential',
    'product_innovation': 'Product Innovation',
    'product_market_fit': 'Product Market Fit',
    'founder_competency': 'Founder Competency',
    'founder_idea_fit': 'LLM-Founder-Idea-Fit',
    'pro_overall': 'Pro Overall Score',
    'basic_overall': 'Basic Overall Score',
    'founder_segmentation': 'Founder Segmentation',
    'top_founder_idea_fit': 'Statistical-Founder-Idea-Fit'
}

PREDICTION_PATHS = {
    'pro_prediction': ('final_analysis_pro_structured', 'outcome'),
    'basic_prediction': ('basic_analysis_structured', 'outcome')
}

def extract_data(experiment_file_path):
    """Loads data from the SSFF experiment JSON file and extracts scores and predictions."""
    logging.info(f"Loading data from {experiment_file_path}...")
    try:
        with open(experiment_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Experiment file not found at {experiment_file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {experiment_file_path}")
        return None

    extracted_records = []
    for record in data:
        if not isinstance(record, dict):
            logging.warning(f"Skipping non-dictionary record: {record}")
            continue
            
        res = {'org_uuid': record.get('org_uuid', pd.NA)}

        for score_name, path_tuple in SCORES_TO_ANALYZE.items():
            main_key, sub_key = path_tuple
            value = pd.NA
            if sub_key is None: # Direct key
                value = record.get(main_key, pd.NA)
            elif main_key in record and isinstance(record[main_key], dict):
                value = record[main_key].get(sub_key, pd.NA)
            
            if pd.notna(value):
                try:
                    res[score_name] = float(value)
                except (ValueError, TypeError):
                    logging.warning(f"Could not convert score '{score_name}' to float for org {res['org_uuid']}. Value: {value}. Setting to NA.")
                    res[score_name] = pd.NA
            else:
                res[score_name] = pd.NA

        for pred_name, path_tuple in PREDICTION_PATHS.items():
            main_key, sub_key = path_tuple
            raw_value = pd.NA
            
            if main_key in record and isinstance(record[main_key], dict):
                raw_value = record[main_key].get(sub_key, pd.NA)

            if pd.isna(raw_value):
                res[pred_name] = pd.NA
            elif isinstance(raw_value, str):
                # Specific handling for basic_prediction as it was problematic
                if pred_name == 'basic_prediction':
                    stripped_lower_value = raw_value.strip().lower()
                    if stripped_lower_value.startswith('invest'):
                        res[pred_name] = 'Invest'
                    elif stripped_lower_value.startswith('hold'):
                        res[pred_name] = 'Hold'
                    else:
                        logging.warning(f"Unexpected string value for basic_prediction '{str(raw_value)[:100]}...' for org {res['org_uuid']}. Setting to NA.")
                        res[pred_name] = pd.NA
                # Add similar handling for pro_prediction
                elif pred_name == 'pro_prediction':
                    stripped_lower_value = raw_value.strip().lower()
                    if stripped_lower_value.startswith('invest'):
                        res[pred_name] = 'Invest'
                    elif stripped_lower_value.startswith('hold'):
                        res[pred_name] = 'Hold'
                    else:
                        logging.warning(f"Unexpected string value for pro_prediction '{str(raw_value)[:100]}...' for org {res['org_uuid']}. Setting to NA.")
                        res[pred_name] = pd.NA
                else: # For any other future predictions, assume they are clean for now
                    res[pred_name] = raw_value.strip()
            else: # If raw_value is not string or NA
                logging.warning(f"Non-string, non-NA value for {pred_name} '{str(raw_value)[:100]}...' for org {res['org_uuid']}. Setting to NA.")
                res[pred_name] = pd.NA
            
        # Extract and map the ground truth label
        label_value = record.get('label', pd.NA)
        if pd.notna(label_value):
            try:
                label_numeric = int(label_value)
                if label_numeric == 1:
                    res['actual_outcome'] = 'Actually Successful'
                elif label_numeric == 0:
                    res['actual_outcome'] = 'Actually Unsuccessful'
                else:
                    res['actual_outcome'] = 'Actual Outcome Unknown' # Or pd.NA
                    logging.warning(f"Unexpected numeric label value '{label_value}' for org {res['org_uuid']}. Setting actual_outcome to Unknown.")
            except (ValueError, TypeError):
                res['actual_outcome'] = 'Actual Outcome Unknown' # Or pd.NA
                logging.warning(f"Could not convert label '{label_value}' to int for org {res['org_uuid']}. Setting actual_outcome to Unknown.")
        else:
            res['actual_outcome'] = pd.NA # Or 'Actual Outcome Unknown' if preferred for plotting
            
        extracted_records.append(res)

    df = pd.DataFrame(extracted_records)
    logging.info(f"Successfully extracted {len(df)} records into a DataFrame.")
    return df

def plot_grand_score_comparison(df, score_cols_dict, output_dir):
    """Plots a grand comparison of average scores across Pro/Basic Invest/Hold conditions."""
    logging.info("Generating grand score comparison plot...")
    
    plot_data_list = []
    internal_score_names = list(score_cols_dict.keys()) # Internal keys

    for score_col_internal in internal_score_names:
        display_score_name = SCORE_DISPLAY_NAMES.get(score_col_internal, score_col_internal)
        if score_col_internal not in df.columns or pd.to_numeric(df[score_col_internal], errors='coerce').dropna().empty:
            logging.warning(f"Skipping score '{display_score_name}' (key: {score_col_internal}) in grand comparison due to missing or non-numeric data.")
            continue

        # Ensure score column is numeric for calculations
        df[score_col_internal] = pd.to_numeric(df[score_col_internal], errors='coerce')

        # Pro Predictions
        if 'pro_prediction' in df.columns:
            pro_invest_scores = df[df['pro_prediction'] == 'Invest'][score_col_internal].dropna()
            pro_hold_scores = df[df['pro_prediction'] == 'Hold'][score_col_internal].dropna()
            
            if not pro_invest_scores.empty:
                plot_data_list.append({'score_name': display_score_name, 'mean_score': pro_invest_scores.mean(), 
                                       'condition': 'Pro: Invest', 'count': len(pro_invest_scores)})
            if not pro_hold_scores.empty:
                plot_data_list.append({'score_name': display_score_name, 'mean_score': pro_hold_scores.mean(), 
                                       'condition': 'Pro: Hold', 'count': len(pro_hold_scores)})
        # Basic Predictions
        if 'basic_prediction' in df.columns:
            basic_invest_scores = df[df['basic_prediction'] == 'Invest'][score_col_internal].dropna()
            basic_hold_scores = df[df['basic_prediction'] == 'Hold'][score_col_internal].dropna()

            if not basic_invest_scores.empty:
                plot_data_list.append({'score_name': display_score_name, 'mean_score': basic_invest_scores.mean(),
                                       'condition': 'Basic: Invest', 'count': len(basic_invest_scores)})
            if not basic_hold_scores.empty:
                plot_data_list.append({'score_name': display_score_name, 'mean_score': basic_hold_scores.mean(),
                                       'condition': 'Basic: Hold', 'count': len(basic_hold_scores)})
    
    if not plot_data_list:
        logging.warning("No data available for the grand score comparison plot. Skipping.")
        return

    grand_plot_df = pd.DataFrame(plot_data_list)
    # Sort by display score name for consistent plotting order if needed
    grand_plot_df['score_name'] = pd.Categorical(grand_plot_df['score_name'], categories=[SCORE_DISPLAY_NAMES.get(k, k) for k in internal_score_names], ordered=True)
    grand_plot_df.sort_values('score_name', inplace=True)


    g = sns.catplot(x='score_name', y='mean_score', hue='condition', data=grand_plot_df, 
                    kind='bar', errorbar=('ci', 95), capsize=.05,
                    palette={'Pro: Invest': '#1f77b4', 'Pro: Hold': '#aec7e8', 
                             'Basic: Invest': '#ff7f0e', 'Basic: Hold': '#ffbb78'},
                    legend=False, aspect=max(1.5, len(internal_score_names)/4), height=6) 

    g.set_xticklabels(rotation=45, ha='right', fontsize=10)
    g.set_axis_labels("Score Category", "Mean Score (95% CI)", fontsize=12)
    g.fig.suptitle('Grand Comparison of Mean Scores by Prediction Outcome', fontsize=16, y=1.03) # y adjusted for suptitle
    
    # Adjust legend position and layout
    handles, labels = g.ax.get_legend_handles_labels()
    if handles: # Only add legend if there are handles
        g.ax.legend(handles, labels, title='Condition', loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, title_fontsize=11)
    
    g.fig.subplots_adjust(bottom=0.2, right=0.85) # Adjust subplot to make space for labels and legend

    plot_path = os.path.join(output_dir, 'grand_score_comparison.png')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        logging.info(f"Saved grand score comparison plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save grand score comparison plot {plot_path}: {e}")
    plt.close('all')


def plot_prediction_consistency(df, output_dir):
    """Plots the consistency between Pro and Basic predictions."""
    logging.info("Generating prediction consistency plot...")
    if 'pro_prediction' not in df.columns or 'basic_prediction' not in df.columns:
        logging.warning("Pro or Basic prediction columns not found. Skipping prediction consistency plot.")
        return

    # Ensure consistent category order for crosstab
    pro_categories = sorted(df['pro_prediction'].dropna().unique().tolist())
    basic_categories = sorted(df['basic_prediction'].dropna().unique().tolist())

    if not pro_categories or not basic_categories:
        logging.warning("Not enough unique values in prediction columns for consistency plot.")
        return

    # Create a DataFrame with all combinations to ensure all categories are present in the heatmap
    all_orgs = df[['org_uuid', 'pro_prediction', 'basic_prediction']].copy()
    all_orgs['pro_prediction'] = pd.Categorical(all_orgs['pro_prediction'], categories=pro_categories, ordered=True)
    all_orgs['basic_prediction'] = pd.Categorical(all_orgs['basic_prediction'], categories=basic_categories, ordered=True)
    
    consistency_counts = pd.crosstab(all_orgs['basic_prediction'], all_orgs['pro_prediction'], dropna=False)
    consistency_normalized = pd.crosstab(all_orgs['basic_prediction'], all_orgs['pro_prediction'], normalize='all', dropna=False)


    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(consistency_counts, annot=True, fmt="d", cmap="Blues", linewidths=.5, ax=ax, cbar_kws={'label': 'Count'})
    
    # Add normalized percentages as secondary annotation
    for i in range(consistency_normalized.shape[0]):
        for j in range(consistency_normalized.shape[1]):
            count_val = consistency_counts.iloc[i, j]
            norm_val = consistency_normalized.iloc[i, j]
            ax.text(j + 0.5, i + 0.7, f"({norm_val*100:.1f}%)", 
                    ha="center", va="center", color="grey", fontsize=9)


    ax.set_title('Prediction Consistency: Basic vs. Pro Outcome (Counts and % of Total)', pad=20)
    ax.set_xlabel('Pro Prediction Outcome')
    ax.set_ylabel('Basic Prediction Outcome')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Ensure title and labels fit

    plot_path = os.path.join(output_dir, 'prediction_consistency_heatmap.png')
    try:
        plt.savefig(plot_path)
        logging.info(f"Saved prediction consistency heatmap to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save prediction consistency heatmap {plot_path}: {e}")
    plt.close()


def plot_score_correlation_heatmap(df, score_cols_dict, output_dir):
    """Plots a correlation heatmap for the specified scores."""
    logging.info("Generating score correlation heatmap...")
    internal_score_names = list(score_cols_dict.keys())
    display_names_for_heatmap = [SCORE_DISPLAY_NAMES.get(k, k) for k in internal_score_names]
    
    # Select only the score columns and ensure they are numeric
    score_df = df[internal_score_names].copy()
    for col in internal_score_names:
        if col in score_df.columns:
            score_df[col] = pd.to_numeric(score_df[col], errors='coerce')
    
    score_df.dropna(axis=1, how='all', inplace=True) # Drop columns that are all NA
    score_df.dropna(axis=0, how='any', inplace=True) # Drop rows with any NA for correlation calculation

    if score_df.empty or score_df.shape[1] < 2:
        logging.warning("Not enough valid score data for correlation heatmap. Skipping.")
        return

    correlation_matrix = score_df.corr()
    # Rename columns and index for display in heatmap
    correlation_matrix.columns = [SCORE_DISPLAY_NAMES.get(col, col) for col in correlation_matrix.columns]
    correlation_matrix.index = [SCORE_DISPLAY_NAMES.get(idx, idx) for idx in correlation_matrix.index]


    plt.figure(figsize=(max(10, len(internal_score_names)*0.9), max(8, len(internal_score_names)*0.8)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Scores', pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(pad=1.5)

    plot_path = os.path.join(output_dir, 'score_correlation_heatmap.png')
    try:
        plt.savefig(plot_path)
        logging.info(f"Saved score correlation heatmap to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save score correlation heatmap {plot_path}: {e}")
    plt.close()


def plot_conditional_overall_score_distributions(df, output_dir):
    """Plots distributions of overall scores conditioned on the other prediction type."""
    logging.info("Generating conditional overall score distributions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False) # sharey=False as scales might differ
    
    plot_made = False # Flag to track if any plot is made

    # Plot 1: Pro Overall Score distribution by Basic Prediction
    if 'pro_overall' in df.columns and 'basic_prediction' in df.columns:
        df_pro_overall = df[['pro_overall', 'basic_prediction']].copy()
        df_pro_overall['pro_overall'] = pd.to_numeric(df_pro_overall['pro_overall'], errors='coerce')
        df_pro_overall.dropna(subset=['pro_overall', 'basic_prediction'], inplace=True)

        if not df_pro_overall.empty:
            sns.kdeplot(data=df_pro_overall, x='pro_overall', hue='basic_prediction', 
                        fill=True, common_norm=False, ax=axes[0], palette=['#ff7f0e', '#ffbb78', '#2ca02c']) # Added palette
            axes[0].set_title('Distribution of Pro Overall Score\nby Basic Prediction Outcome')
            axes[0].set_xlabel('Pro Overall Score')
            axes[0].set_ylabel('Density')
            plot_made = True
        else:
            axes[0].text(0.5, 0.5, "No data for Pro Overall by Basic Prediction", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
            logging.warning("No data for Pro Overall Score by Basic Prediction plot.")
    else:
        axes[0].text(0.5, 0.5, "Pro Overall or Basic Prediction data missing", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        logging.warning("Pro Overall Score or Basic Prediction column missing. Skipping corresponding plot.")

    # Plot 2: Basic Overall Score distribution by Pro Prediction
    if 'basic_overall' in df.columns and 'pro_prediction' in df.columns:
        df_basic_overall = df[['basic_overall', 'pro_prediction']].copy()
        df_basic_overall['basic_overall'] = pd.to_numeric(df_basic_overall['basic_overall'], errors='coerce')
        df_basic_overall.dropna(subset=['basic_overall', 'pro_prediction'], inplace=True)

        if not df_basic_overall.empty:
            sns.kdeplot(data=df_basic_overall, x='basic_overall', hue='pro_prediction', 
                        fill=True, common_norm=False, ax=axes[1], palette=['#1f77b4', '#aec7e8', '#d62728']) # Added palette
            axes[1].set_title('Distribution of Basic Overall Score\nby Pro Prediction Outcome')
            axes[1].set_xlabel('Basic Overall Score')
            axes[1].set_ylabel('Density') # Only show Y-label if there is data
            plot_made = True
        else:
            axes[1].text(0.5, 0.5, "No data for Basic Overall by Pro Prediction", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
            logging.warning("No data for Basic Overall Score by Pro Prediction plot.")

    else:
        axes[1].text(0.5, 0.5, "Basic Overall or Pro Prediction data missing", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        logging.warning("Basic Overall Score or Pro Prediction column missing. Skipping corresponding plot.")

    if not plot_made:
        logging.warning("No conditional overall score distribution plots could be generated due to missing data. Skipping save.")
        plt.close()
        return

    fig.suptitle('Conditional Overall Score Distributions', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap

    plot_path = os.path.join(output_dir, 'conditional_overall_score_distributions.png')
    try:
        plt.savefig(plot_path)
        logging.info(f"Saved conditional overall score distributions plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save conditional distributions plot {plot_path}: {e}")
    plt.close()


def plot_score_distribution(df, score_column, output_dir):
    """Plots and saves the distribution of a given score."""
    if score_column not in df.columns or df[score_column].dropna().empty:
        logging.warning(f"Score column '{SCORE_DISPLAY_NAMES.get(score_column, score_column)}' (key: {score_column}) not found or empty. Skipping distribution plot.")
        return

    plt.figure(figsize=(10, 6))
    # Convert to numeric, coercing errors, then drop NAs for plotting
    numeric_scores = pd.to_numeric(df[score_column], errors='coerce').dropna()
    display_score_name = SCORE_DISPLAY_NAMES.get(score_column, score_column)
    if numeric_scores.empty:
        logging.warning(f"No valid numeric data for score '{display_score_name}' (key: {score_column}) after coercion. Skipping distribution plot.")
        plt.close() # Close the figure if no data
        return
        
    sns.histplot(numeric_scores, kde=True, bins=20)
    plt.title(f'Distribution of {display_score_name}', fontsize=15)
    plt.xlabel(display_score_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    
    plot_filename = f'{score_column}_distribution.png' # Keep internal name for filename consistency
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        logging.info(f"Saved distribution plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_path}: {e}")
    plt.close()

def perform_group_statistical_tests(df, score_column, group_column, output_dir_stats):
    """
    Performs Mann-Whitney U test for a score between two groups and appends to summary file.
    (Formerly part of analyze_group_differences)
    """
    if score_column not in df.columns or df[score_column].dropna().empty:
        logging.warning(f"Score column '{SCORE_DISPLAY_NAMES.get(score_column, score_column)}' (key: {score_column}) not found or empty for stats. Skipping.")
        return
    if group_column not in df.columns or df[group_column].dropna().empty:
        logging.warning(f"Group column '{group_column}' not found or empty for stats. Skipping.")
        return

    # Ensure score_column is numeric and drop NAs for this specific analysis
    df_analysis = df[[score_column, group_column]].copy()
    df_analysis[score_column] = pd.to_numeric(df_analysis[score_column], errors='coerce')
    df_analysis.dropna(subset=[score_column, group_column], inplace=True)
    
    display_score_name = SCORE_DISPLAY_NAMES.get(score_column, score_column)

    if df_analysis.empty:
        logging.warning(f"No valid data for statistical test of '{display_score_name}' (key: {score_column}) by '{group_column}'.")
        return
    
    # Statistical Test (Mann-Whitney U)
    groups = df_analysis[group_column].unique()
    stat_summary_lines = []

    # Ensure groups are sorted for consistent output if applicable, though order doesn't affect test for 2 groups.
    groups = sorted([str(g) for g in groups if pd.notna(g)])


    if len(groups) == 2:
        group1_data = df_analysis[df_analysis[group_column] == groups[0]][score_column]
        group2_data = df_analysis[df_analysis[group_column] == groups[1]][score_column]

        if len(group1_data) < 2 or len(group2_data) < 2: # Min samples for a meaningful test
            logging.warning(f"Not enough data in one or both groups for {display_score_name} (key: {score_column}) by {group_column} for statistical test.")
            stat_summary_lines.append(f"Insufficient data for test between {groups[0]} and {groups[1]} for score {display_score_name} (key: {score_column}).")
        else:
            try:
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                stat_summary_lines.append(f"Mann-Whitney U test for {display_score_name} (key: {score_column}) between '{groups[0]}' (N={len(group1_data)}, Mean={group1_data.mean():.2f}) and '{groups[1]}' (N={len(group2_data)}, Mean={group2_data.mean():.2f}):")
                stat_summary_lines.append(f"  Statistic: {stat:.2f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    stat_summary_lines.append("  Difference is statistically significant (p < 0.05).")
                else:
                    stat_summary_lines.append("  Difference is not statistically significant (p >= 0.05).")
            except Exception as e:
                logging.error(f"Error during Mann-Whitney U test for {display_score_name} (key: {score_column}) by {group_column}: {e}")
                stat_summary_lines.append(f"Error during statistical test for {display_score_name} (key: {score_column}) by {group_column}.")
    elif len(groups) < 2:
        stat_summary_lines.append(f"Only one group ('{groups[0]}' if len(groups) > 0 else 'None') found for {display_score_name} (key: {score_column}) by {group_column}. Cannot perform two-group test.")
    else: 
        stat_summary_lines.append(f"More than two groups found for {group_column} ({groups}). Mann-Whitney U is for two groups. Consider ANOVA or Kruskal-Wallis for multi-group comparison for score {display_score_name} (key: {score_column}).")
    
    stat_summary = "\n".join(stat_summary_lines)
    if stat_summary: # Only log and write if there's something to summarize
        logging.info(f"Statistical summary for {display_score_name} (key: {score_column}) by {group_column}:\n{stat_summary}")
    
        stats_file_path = os.path.join(output_dir_stats, 'statistical_tests_summary.txt')
        try:
            with open(stats_file_path, 'a') as f:
                f.write(f"--- Analysis for {display_score_name} (key: {score_column}) by {group_column} ---\n")
                f.write(stat_summary)
                f.write("\n\n")
        except Exception as e:
            logging.error(f"Failed to write stats to {stats_file_path}: {e}")


def analyze_group_differences(df, score_column, group_column, output_dir_plots, output_dir_stats):
    """
    Calculates average scores for groups, plots grouped bar chart with error bars,
    and performs Mann-Whitney U test.
    NOTE: Plotting part is now superseded by plot_grand_score_comparison. This function
    will now primarily focus on calling the statistical test.
    The bar plot generation here can be considered redundant if grand comparison is active.
    """
    # Statistical tests are now handled by perform_group_statistical_tests
    perform_group_statistical_tests(df, score_column, group_column, output_dir_stats)

    # The original plotting part is left here but commented out or can be removed
    # as plot_grand_score_comparison is more comprehensive.
    # If specific individual plots are still desired, this section could be enabled.
    
    # if score_column not in df.columns or df[score_column].dropna().empty:
    #     logging.warning(f"Score column '{score_column}' not found or empty. Skipping group analysis plot.")
    #     return
    # if group_column not in df.columns or df[group_column].dropna().empty:
    #     logging.warning(f"Group column '{group_column}' not found or empty. Skipping group analysis plot.")
    #     return

    # df_analysis = df[[score_column, group_column]].copy()
    # df_analysis[score_column] = pd.to_numeric(df_analysis[score_column], errors='coerce')
    # df_analysis.dropna(subset=[score_column, group_column], inplace=True)

    # if df_analysis.empty:
    #     logging.warning(f"No valid data for group analysis plot of '{score_column}' by '{group_column}'.")
    #     return
    
    # # Grouped Bar Chart (This is the part that is now redundant due to plot_grand_score_comparison)
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=group_column, y=score_column, data=df_analysis, estimator=np.mean, errorbar=('ci', 95), capsize=.1, palette=['#66b3ff','#ff9999','#99ff99'])
    # plt.title(f'Average {score_column} by {group_column} (95% CI)')
    # plt.xlabel(group_column)
    # plt.ylabel(f'Average {score_column}')
    
    # plot_path = os.path.join(output_dir_plots, f'{score_column}_by_{group_column}_avg_individual.png') # Renamed to avoid conflict
    # try:
    #     plt.savefig(plot_path)
    #     logging.info(f"Saved individual grouped bar plot to {plot_path}")
    # except Exception as e:
    #     logging.error(f"Failed to save plot {plot_path}: {e}")
    # plt.close()
    pass # End of analyze_group_differences - now primarily a wrapper or can be fully replaced in main loop


def plot_grand_score_comparison_by_actual(df, score_cols_dict, output_dir):
    """Plots a grand comparison of average scores based on actual startup success/failure."""
    logging.info("Generating grand score comparison plot by actual outcome...")
    
    if 'actual_outcome' not in df.columns or df['actual_outcome'].dropna().empty:
        logging.warning("'actual_outcome' column not found or empty. Skipping grand comparison by actual outcome.")
        return

    plot_data_list = []
    internal_score_names = list(score_cols_dict.keys()) # Internal keys

    for score_col_internal in internal_score_names:
        display_score_name = SCORE_DISPLAY_NAMES.get(score_col_internal, score_col_internal)
        if score_col_internal not in df.columns or pd.to_numeric(df[score_col_internal], errors='coerce').dropna().empty:
            logging.warning(f"Skipping score '{display_score_name}' (key: {score_col_internal}) in actual outcome comparison due to missing or non-numeric data.")
            continue

        df[score_col_internal] = pd.to_numeric(df[score_col_internal], errors='coerce')

        actual_successful_scores = df[df['actual_outcome'] == 'Actually Successful'][score_col_internal].dropna()
        actual_unsuccessful_scores = df[df['actual_outcome'] == 'Actually Unsuccessful'][score_col_internal].dropna()
        
        if not actual_successful_scores.empty:
            plot_data_list.append({
                'score_name': display_score_name, 
                'mean_score': actual_successful_scores.mean(), 
                'condition': 'Actually Successful', 
                'count': len(actual_successful_scores)
            })
        if not actual_unsuccessful_scores.empty:
            plot_data_list.append({
                'score_name': display_score_name, 
                'mean_score': actual_unsuccessful_scores.mean(), 
                'condition': 'Actually Unsuccessful', 
                'count': len(actual_unsuccessful_scores)
            })
    
    if not plot_data_list:
        logging.warning("No data available for the grand score comparison by actual outcome plot. Skipping.")
        return

    grand_plot_df = pd.DataFrame(plot_data_list)
    # Sort by display score name for consistent plotting order
    grand_plot_df['score_name'] = pd.Categorical(grand_plot_df['score_name'], categories=[SCORE_DISPLAY_NAMES.get(k, k) for k in internal_score_names], ordered=True)
    grand_plot_df.sort_values('score_name', inplace=True)

    g = sns.catplot(x='score_name', y='mean_score', hue='condition', data=grand_plot_df, 
                    kind='bar', errorbar=('ci', 95), capsize=.05,
                    palette={'Actually Successful': '#2ca02c', 'Actually Unsuccessful': '#d62728'},
                    legend=False, aspect=max(1.5, len(internal_score_names)/5), height=6) # Adjusted aspect ratio slightly

    g.set_xticklabels(rotation=45, ha='right', fontsize=10)
    g.set_axis_labels("Score Category", "Mean Score (95% CI)", fontsize=12)
    g.fig.suptitle('Grand Comparison of Mean Scores by Actual Outcome', fontsize=16, y=1.03)
    
    handles, labels = g.ax.get_legend_handles_labels()
    if handles:
        g.ax.legend(handles, labels, title='Actual Outcome', loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, title_fontsize=11)
    
    g.fig.subplots_adjust(bottom=0.2, right=0.83) # Adjust right margin for legend

    plot_path = os.path.join(output_dir, 'grand_score_comparison_by_actual_outcome.png')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        logging.info(f"Saved grand score comparison by actual outcome plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save grand score comparison by actual outcome plot {plot_path}: {e}")
    plt.close('all')

def plot_overall_scores_by_actual_distribution(df, output_dir):
    """Plots distributions of Pro and Basic overall scores conditioned on actual startup success/failure."""
    logging.info("Generating overall score distributions by actual outcome...")

    if 'actual_outcome' not in df.columns or df['actual_outcome'].dropna().empty:
        logging.warning("'actual_outcome' column not found or empty. Skipping overall score distributions by actual outcome.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    plot_made_count = 0

    # Plot 1: Pro Overall Score distribution by Actual Outcome
    if 'pro_overall' in df.columns:
        df_pro_actual = df[['pro_overall', 'actual_outcome']].copy()
        df_pro_actual['pro_overall'] = pd.to_numeric(df_pro_actual['pro_overall'], errors='coerce')
        df_pro_actual.dropna(subset=['pro_overall', 'actual_outcome'], inplace=True)
        pro_overall_display_name = SCORE_DISPLAY_NAMES.get('pro_overall', 'Pro Overall Score')

        if not df_pro_actual.empty and len(df_pro_actual['actual_outcome'].unique()) > 0:
            sns.kdeplot(data=df_pro_actual, x='pro_overall', hue='actual_outcome', 
                        fill=True, common_norm=False, ax=axes[0], 
                        palette={'Actually Successful': '#2ca02c', 'Actually Unsuccessful': '#d62728', 'Actual Outcome Unknown': '#7f7f7f'})
            axes[0].set_title(f'Distribution of {pro_overall_display_name}\nby Actual Startup Outcome', fontsize=14)
            axes[0].set_xlabel(pro_overall_display_name, fontsize=12)
            plot_made_count += 1
        else:
            axes[0].text(0.5, 0.5, "No data for Pro Overall by Actual Outcome", ha='center', va='center', transform=axes[0].transAxes)
            logging.warning("No data for Pro Overall Score by Actual Outcome plot.")
    else:
        axes[0].text(0.5, 0.5, "Pro Overall data missing", ha='center', va='center', transform=axes[0].transAxes)
        logging.warning("Pro Overall Score column missing. Skipping corresponding actual outcome plot.")

    # Plot 2: Basic Overall Score distribution by Actual Outcome
    if 'basic_overall' in df.columns:
        df_basic_actual = df[['basic_overall', 'actual_outcome']].copy()
        df_basic_actual['basic_overall'] = pd.to_numeric(df_basic_actual['basic_overall'], errors='coerce')
        df_basic_actual.dropna(subset=['basic_overall', 'actual_outcome'], inplace=True)
        basic_overall_display_name = SCORE_DISPLAY_NAMES.get('basic_overall', 'Basic Overall Score')

        if not df_basic_actual.empty and len(df_basic_actual['actual_outcome'].unique()) > 0:
            sns.kdeplot(data=df_basic_actual, x='basic_overall', hue='actual_outcome', 
                        fill=True, common_norm=False, ax=axes[1], 
                        palette={'Actually Successful': '#2ca02c', 'Actually Unsuccessful': '#d62728', 'Actual Outcome Unknown': '#7f7f7f'})
            axes[1].set_title(f'Distribution of {basic_overall_display_name}\nby Actual Startup Outcome', fontsize=14)
            axes[1].set_xlabel(basic_overall_display_name, fontsize=12)
            plot_made_count += 1
        else:
            axes[1].text(0.5, 0.5, "No data for Basic Overall by Actual Outcome", ha='center', va='center', transform=axes[1].transAxes)
            logging.warning("No data for Basic Overall Score by Actual Outcome plot.")
    else:
        axes[1].text(0.5, 0.5, "Basic Overall data missing", ha='center', va='center', transform=axes[1].transAxes)
        logging.warning("Basic Overall Score column missing. Skipping corresponding actual outcome plot.")

    if plot_made_count == 0:
        logging.warning("No overall score distribution by actual outcome plots generated. Skipping save.")
        plt.close()
        return

    fig.suptitle('Overall Score Distributions by Actual Startup Outcome', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plot_path = os.path.join(output_dir, 'overall_scores_by_actual_outcome_distribution.png')
    try:
        plt.savefig(plot_path)
        logging.info(f"Saved overall score distributions by actual outcome plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save overall scores by actual outcome distributions plot {plot_path}: {e}")
    plt.close()


def generate_summary_stats_table(df, score_cols_dict, output_dir_stats):
    """Generates and saves a detailed summary statistics table (N, mean, std) for scores under various conditions."""
    logging.info("Generating summary statistics table...")
    
    summary_data_list = []
    internal_score_names = list(score_cols_dict.keys())

    # Define conditions: (grouping_column, group_value, display_label_for_condition)
    conditions_map = {
        'pro_prediction': [('Invest', 'Pro: Invest'), ('Hold', 'Pro: Hold')],
        'basic_prediction': [('Invest', 'Basic: Invest'), ('Hold', 'Basic: Hold')],
        'actual_outcome': [('Actually Successful', 'Actually Successful'), ('Actually Unsuccessful', 'Actually Unsuccessful')]
    }

    for score_col_internal in internal_score_names:
        display_score_name = SCORE_DISPLAY_NAMES.get(score_col_internal, score_col_internal)
        
        if score_col_internal not in df.columns:
            logging.warning(f"Score column '{display_score_name}' (key: {score_col_internal}) not found in DataFrame. Skipping for summary table.")
            continue
        
        # Ensure score column is numeric for calculations, coercing errors to NaN
        df[score_col_internal] = pd.to_numeric(df[score_col_internal], errors='coerce')

        for group_col, condition_list in conditions_map.items():
            if group_col not in df.columns:
                logging.debug(f"Grouping column '{group_col}' not found. Skipping related conditions for score '{display_score_name}'.")
                continue
            
            for group_val, condition_label in condition_list:
                # Filter data for the current condition
                # Ensure NaNs in group_col are handled if group_val is also NaN (though not typical here)
                if pd.isna(group_val): 
                    conditioned_scores = df[df[group_col].isna()][score_col_internal].dropna()
                else:
                    conditioned_scores = df[df[group_col] == group_val][score_col_internal].dropna()

                if not conditioned_scores.empty:
                    mean_score = conditioned_scores.mean()
                    std_score = conditioned_scores.std()
                    count = len(conditioned_scores)
                    summary_data_list.append({
                        'Score Name': display_score_name,
                        'Condition': condition_label,
                        'N': count,
                        'Mean': mean_score,
                        'Std': std_score
                    })
                else:
                    summary_data_list.append({
                        'Score Name': display_score_name,
                        'Condition': condition_label,
                        'N': 0,
                        'Mean': np.nan, # Use np.nan for consistency with pandas
                        'Std': np.nan
                    })
                    logging.debug(f"No data for score '{display_score_name}' under condition '{condition_label}'. N=0.")
                
    if not summary_data_list:
        logging.warning("No data available to build the summary statistics table. Skipping CSV generation.")
        return

    summary_df = pd.DataFrame(summary_data_list)
    
    # Order Score Name column based on SCORES_TO_ANALYZE for consistent output
    ordered_score_names = [SCORE_DISPLAY_NAMES.get(k, k) for k in internal_score_names if k in df.columns and df[k].notna().any()]
    summary_df['Score Name'] = pd.Categorical(summary_df['Score Name'], categories=ordered_score_names, ordered=True)
    
    # Sort the table for better readability
    summary_df.sort_values(by=['Score Name', 'Condition'], inplace=True)

    stats_table_path = os.path.join(output_dir_stats, 'detailed_score_statistics.csv')
    try:
        summary_df.to_csv(stats_table_path, index=False, float_format='%.3f')
        logging.info(f"Saved detailed score statistics table to {stats_table_path}")
    except Exception as e:
        logging.error(f"Failed to save detailed score statistics table {stats_table_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SSFF experiment JSON file for quantitative scores.")
    parser.add_argument("experiment_file_path", type=str, help="Path to the SSFF experiment JSON file.")
    parser.add_argument("output_dir", type=str, help="Directory to save plots and analysis results.")
    
    args = parser.parse_args()

    output_plots_dir = os.path.join(args.output_dir, "plots")
    output_stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(output_plots_dir, exist_ok=True)
    os.makedirs(output_stats_dir, exist_ok=True)
    
    stats_summary_file = os.path.join(output_stats_dir, 'statistical_tests_summary.txt')
    if os.path.exists(stats_summary_file):
        try:
            os.remove(stats_summary_file) # Clear summary stats file for a fresh run
            logging.info(f"Removed existing stats summary file: {stats_summary_file}")
        except OSError as e:
            logging.error(f"Error removing existing stats file {stats_summary_file}: {e}")


    df = extract_data(args.experiment_file_path)

    if df is None or df.empty:
        logging.error("No data extracted. Exiting.")
        return

    logging.info("Starting analysis of scores...")

    # New consolidated plotting functions
    plot_grand_score_comparison(df, SCORES_TO_ANALYZE, output_plots_dir)
    plot_prediction_consistency(df, output_plots_dir)
    plot_score_correlation_heatmap(df, SCORES_TO_ANALYZE, output_plots_dir)
    plot_conditional_overall_score_distributions(df, output_plots_dir)

    # --- New plots based on actual outcomes ---
    logging.info("\n--- Generating plots based on Actual Startup Outcomes ---")
    plot_grand_score_comparison_by_actual(df, SCORES_TO_ANALYZE, output_plots_dir)
    plot_overall_scores_by_actual_distribution(df, output_plots_dir)
    # --- End new plots ---

    # --- Generate Detailed Statistics Table ---
    logging.info("\n--- Generating Detailed Score Statistics Table ---")
    generate_summary_stats_table(df, SCORES_TO_ANALYZE, output_stats_dir)
    # --- End Detailed Statistics Table ---

    # Individual score distributions (optional, can be commented out if redundant)
    # logging.info("\n--- Generating individual score distributions ---")
    # for score_col in SCORES_TO_ANALYZE.keys():
    #     plot_score_distribution(df, score_col, output_plots_dir)

    # Perform statistical tests (separated from plotting)
    if 'pro_prediction' in df.columns:
        logging.info("\n--- Performing statistical tests for scores based on Pro Prediction ---")
        for score_col_internal in SCORES_TO_ANALYZE.keys():
            perform_group_statistical_tests(df, score_col_internal, 'pro_prediction', output_stats_dir)
    else:
        logging.warning("Column 'pro_prediction' not found. Skipping statistical tests for Pro predictions.")

    if 'basic_prediction' in df.columns:
        logging.info("\n--- Performing statistical tests for scores based on Basic Prediction ---")
        relevant_basic_score_keys = [s_key for s_key in SCORES_TO_ANALYZE.keys() if s_key != 'pro_overall'] # Exclude pro_overall for basic prediction tests
        for score_col_internal in relevant_basic_score_keys:
            perform_group_statistical_tests(df, score_col_internal, 'basic_prediction', output_stats_dir)
    else:
        logging.warning("Column 'basic_prediction' not found. Skipping statistical tests for Basic predictions.")
        
    logging.info(f"Analysis complete. Check the output directory '{args.output_dir}' for plots and stats summary.")

if __name__ == "__main__":
    main() 