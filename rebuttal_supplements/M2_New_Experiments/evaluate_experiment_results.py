import json
import argparse
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import logging
import numpy as np # Import numpy for F-beta calculation if sklearn doesn't have it directly or for custom calcs
import matplotlib.pyplot as plt # Added import
import seaborn as sns # Added import

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

def f_beta_score(precision, recall, beta):
    """Calculates F-beta score."""
    if precision + recall == 0:
        return 0.0
    # Ensure precision and recall are not zero to avoid division by zero if beta is involved in the denominator's parts
    if precision == 0 and recall == 0: # Technically covered by precision + recall == 0, but for clarity
        return 0.0
    
    # If beta_squared * precision + recall is zero (can happen if beta=0 and recall=0, or precision=0 and beta is high)
    # and the numerator is also zero (recall=0), then F-beta is 0.
    # If recall is non-zero but beta_squared * precision + recall is zero, this implies an issue or precision is zero and beta makes its term vanish.
    # F-beta formula: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    if denominator == 0:
        return 0.0 # Avoid division by zero; if denominator is 0, numerator is likely 0 too unless recall is non-zero and precision is zero.
    return numerator / denominator

def get_single_prediction(record, file_type_hint, sub_model_type=None):
    """Extracts a single prediction from a record based on file type hint and optional sub_model_type."""
    prediction_str = None
    prediction = None # Default to None
    try:
        if file_type_hint == "baseline" or file_type_hint == "cot":
            prediction_str = record.get("recommendation")
            if prediction_str:
                # Prioritize "unsuccessful" to avoid "successful" in "unsuccessful" being a false positive
                if "unsuccessful" in prediction_str.lower():
                    prediction = 0
                elif "successful" in prediction_str.lower():
                    prediction = 1
                # else: prediction remains None
        elif file_type_hint == "foundergpt":
            score_eta_raw = record.get("final_aggregated_score_eta")
            if score_eta_raw is not None:
                try:
                    score_eta = float(score_eta_raw)
                    if score_eta >= 0.5:  # Threshold from run_experimentation_founder_gpt.py
                        prediction = 1
                    else:
                        prediction = 0
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid or non-convertible 'final_aggregated_score_eta': {score_eta_raw} for record {record.get('original_index', 'N/A')} in FounderGPT file. Error: {e}")
                    prediction = None 
            else:
                logger.warning(f"Missing 'final_aggregated_score_eta' in FounderGPT record {record.get('original_index', 'N/A')}. Cannot determine prediction.")
                prediction = None
        elif file_type_hint == "raise":
            prediction_str = record.get("recommendation") # RAISE uses 'HIGH'/'LOW' or 'Successful'/'Unsuccessful'
            if prediction_str:
                # Prioritize "unsuccessful" and "low" to avoid "successful" in "unsuccessful" being a false positive
                if "unsuccessful" in prediction_str.lower() or "low" in prediction_str.lower():
                    prediction = 0
                elif "successful" in prediction_str.lower() or "high" in prediction_str.lower():
                    prediction = 1
                # else: prediction remains None
        elif file_type_hint == "ssff_nl" or file_type_hint == "ssff_regular":
            details = None # Initialize details
            expected_details_field_name = ""
            expected_outcome_field_name = "outcome" # usually 'outcome'

            if sub_model_type == "Basic":
                expected_details_field_name = "basic_analysis_details" if file_type_hint == "ssff_nl" else "basic_analysis_structured"
                details = record.get(expected_details_field_name)
            elif sub_model_type == "Pro": # For ssff_nl and ssff_regular (Pro version)
                expected_details_field_name = "final_analysis_pro_details" if file_type_hint == "ssff_nl" else "final_analysis_pro_structured"
                details = record.get(expected_details_field_name)
            else:
                logger.error(f"Unknown sub_model_type '{sub_model_type}' for file_type_hint '{file_type_hint}'")
                return None

            if details is None:
                logger.warning(f"Details field '{expected_details_field_name}' not found in record for {file_type_hint} ({sub_model_type})")
                return None

            prediction_str = details.get(expected_outcome_field_name)
            if prediction_str:
                if "invest" in prediction_str.lower():
                    prediction = 1
                elif "hold" in prediction_str.lower(): # or "unsuccessful" / "pass"
                    prediction = 0
                # else: prediction remains None, will be caught by the check below
            
        if prediction_str is None and prediction is None and file_type_hint not in ["foundergpt", "ssff_nl", "ssff_regular"]: # For ssff, prediction_str might be None if outcome is directly 0/1
             logger.warning(f"Prediction string is None for record in {file_type_hint} file. Record: {str(record)[:200]}")

        # If after all checks, prediction is still None, it means we couldn't map it.
        if prediction is None:
            logger.warning(f"Could not determine 0/1 prediction for record. Type: {file_type_hint}, Sub-type: {sub_model_type}, Raw prediction value: '{prediction_str}', Record: {str(record)[:200]}")

    except Exception as e:
        logger.error(f"Error extracting prediction for record: {record}, file_type_hint: {file_type_hint}, sub_model_type: {sub_model_type}. Error: {e}", exc_info=True)
        return None
    
    return prediction

def calculate_and_print_metrics(true_labels, predicted_labels, model_description, filename, results_summary_list):
    if not true_labels or not predicted_labels:
        logger.warning(f"No data to calculate metrics for {model_description}.")
        results_summary_list.append({
            "Model Description": model_description,
            "File": filename,
            "Error": "No data for metrics"
        })
        return

    if len(set(true_labels)) < 2 and len(true_labels) > 0 :
        logger.warning(f"Only one class ({set(true_labels).pop()}) present in true labels for {model_description}. Some metrics might be 0 or undefined.")
    if len(set(predicted_labels)) < 2 and len(predicted_labels) > 0:
        logger.warning(f"Only one class ({set(predicted_labels).pop()}) present in predicted labels for {model_description}. Some metrics might be 0 or undefined.")


    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Metrics for class 1 (Successful/Invest)
    precision_1 = precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    recall_1 = recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    f1_1 = f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    f0_5_1 = f_beta_score(precision_1, recall_1, 0.5)

    # Metrics for class 0 (Unsuccessful/Hold)
    precision_0 = precision_score(true_labels, predicted_labels, pos_label=0, zero_division=0)
    recall_0 = recall_score(true_labels, predicted_labels, pos_label=0, zero_division=0)
    f1_0 = f1_score(true_labels, predicted_labels, pos_label=0, zero_division=0)

    # Overall F1 scores
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0] if cm.shape[0]>0 else 0, 
                                                            cm[0,1] if cm.shape[0]>0 and cm.shape[1]>1 else 0, 
                                                            cm[1,0] if cm.shape[0]>1 and cm.shape[1]>0 else 0, 
                                                            cm[1,1] if cm.shape[0]>1 and cm.shape[1]>1 else 0)

    # Predicted Success Ratio
    predicted_0s = predicted_labels.count(0)
    predicted_1s = predicted_labels.count(1)
    total_predictions = len(predicted_labels)
    
    positive_pred_ratio = 0.0
    if total_predictions > 0:
        positive_pred_ratio = predicted_1s / total_predictions

    predicted_success_ratio_str = "N/A"
    if predicted_0s > 0: # Avoid division by zero
        predicted_success_ratio = predicted_1s / predicted_0s
        predicted_success_ratio_str = f"{predicted_1s}/{predicted_0s} ({predicted_success_ratio:.2f})"
    elif predicted_1s > 0:
        predicted_success_ratio_str = f"{predicted_1s}/{predicted_0s} (Inf)" # All predicted as 1
    else: # No predictions (or all were None and skipped)
        predicted_success_ratio_str = f"0/0 (N/A)"


    print(f"\n--- Evaluation Metrics for: {model_description} ---")
    print(f"File: {filename}")
    print(f"Processed Records for Metrics: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predicted Success Ratio (1s/0s): {predicted_success_ratio_str}")
    print(f"Positive Prediction Ratio (Predicted 1s / Total Predictions): {positive_pred_ratio:.4f}")
    print("\nMetrics for Class 0 (Unsuccessful/Hold):")
    print(f"  Precision: {precision_0:.4f}")
    print(f"  Recall (Specificity for class 1): {recall_0:.4f}")
    print(f"  F1-score: {f1_0:.4f}")
    print("\nMetrics for Class 1 (Successful/Invest):")
    print(f"  Precision: {precision_1:.4f}")
    print(f"  Recall (Sensitivity): {recall_1:.4f}")
    print(f"  F1-score: {f1_1:.4f}")
    print(f"  F0.5-score: {f0_5_1:.4f}") # Added F0.5 score for class 1

    print("\nOverall F1 Scores:")
    print(f"  Macro-Averaged F1-score: {f1_macro:.4f}")
    print(f"  Weighted-Averaged F1-score: {f1_weighted:.4f}")
    
    print("\nConfusion Matrix (labels=[0, 1] -> [Unsuccessful, Successful]):")
    print(f"  True Negatives (TN) [Pred Unsuccessful, True Unsuccessful]: {tn}")
    print(f"  False Positives (FP) [Pred Successful, True Unsuccessful]: {fp}")
    print(f"  False Negatives (FN) [Pred Unsuccessful, True Successful]: {fn}")
    print(f"  True Positives (TP) [Pred Successful, True Successful]: {tp}")
    print("-------------------------")

    results_summary_list.append({
        "Model Description": model_description,
        "File": filename,
        "Processed Records": len(true_labels),
        "Accuracy": accuracy,
        "Predicted Success Ratio": predicted_success_ratio_str,
        "Positive Prediction Ratio": positive_pred_ratio,
        "Precision (Class 0)": precision_0,
        "Recall (Class 0)": recall_0,
        "F1-score (Class 0)": f1_0,
        "Precision (Class 1)": precision_1,
        "Recall (Class 1)": recall_1,
        "F1-score (Class 1)": f1_1,
        "F0.5-score (Class 1)": f0_5_1, # Added F0.5 score to CSV
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    })

def plot_key_metrics(df_summary, plot_dir):
    """Generates and saves bar plots for key performance metrics."""
    if df_summary.empty:
        logger.warning("Summary DataFrame is empty. Skipping plot generation.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Saving metric comparison plots to: {plot_dir}")

    metrics_to_plot = {
        "Accuracy": "Accuracy Comparison",
        "F1-score (Class 1)": "F1-score (Successful/Invest) Comparison",
        "Precision (Class 1)": "Precision (Successful/Invest) Comparison",
        "Recall (Class 1)": "Recall (Successful/Invest) Comparison",
        "F0.5-score (Class 1)": "F0.5-score (Successful/Invest) Comparison",
        "Macro F1": "Macro F1-score Comparison",
        "Positive Prediction Ratio": "Positive Prediction Ratio Comparison"
    }

    # Sort by Model Description for consistent plot order, if desired, or by a metric
    # df_summary.sort_values(by="Model Description", inplace=True)

    num_models = len(df_summary["Model Description"].unique())
    fig_width = max(10, num_models * 0.8) # Adjust width based on number of models

    for metric_col, plot_title in metrics_to_plot.items():
        if metric_col not in df_summary.columns:
            logger.warning(f"Metric column '{metric_col}' not found in summary. Skipping '{plot_title}' plot.")
            continue
        
        # Ensure metric column is numeric and handle potential NAs for plotting
        df_plot = df_summary.copy()
        df_plot[metric_col] = pd.to_numeric(df_plot[metric_col], errors='coerce')
        df_plot.dropna(subset=[metric_col], inplace=True)

        if df_plot.empty:
            logger.warning(f"No valid data for metric '{metric_col}' after coercion/NA drop. Skipping '{plot_title}' plot.")
            continue

        plt.figure(figsize=(fig_width, 6))
        try:
            barplot = sns.barplot(x="Model Description", y=metric_col, data=df_plot, palette="viridis")
            
            # Add value labels on top of bars
            for p in barplot.patches:
                barplot.annotate(format(p.get_height(), '.3f'), 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha = 'center', va = 'center', 
                                 xytext = (0, 9), 
                                 textcoords = 'offset points')

            plt.title(plot_title, fontsize=15)
            plt.xlabel("Model / Analysis Type", fontsize=12)
            plt.ylabel(metric_col, fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.ylim(0, 1.05) # Assuming metrics are between 0 and 1
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_filename = f"{metric_col.replace('(', '').replace(')', '').replace(' ', '_').lower()}_comparison.png"
            plt.savefig(os.path.join(plot_dir, plot_filename))
            logger.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"Error generating plot for {metric_col}: {e}")
        finally:
            plt.close() # Close the figure to free memory

def evaluate_results(filepath, all_results_summary, file_type_hint=None):
    logger.info(f"Evaluating results from: {filepath}")
    filename = os.path.basename(filepath)

    if file_type_hint is None:
        if "baseline_cot" in filename or "cot_" in filename: # Adjusted hint for cot
            file_type_hint = "cot"
        elif "baseline" in filename:
            file_type_hint = "baseline"
        elif "foundergpt" in filename:
            file_type_hint = "foundergpt"
        elif "raise" in filename:
            file_type_hint = "raise"
        elif "ssff_nl" in filename:
            file_type_hint = "ssff_nl"
        elif "ssff_regular" in filename:
            file_type_hint = "ssff_regular"
        else:
            logger.error(f"Could not determine file type for {filename}. Skipping.")
            return

    logger.info(f"Determined file type hint: '{file_type_hint}' for {filename}")

    true_labels_all_records = []
    # First pass to get all true labels for overall dataset distribution
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data] # Handle cases where JSON is a single dict instead of list
            
            for record_idx, record in enumerate(data):
                true_label_raw = record.get("label", record.get("true_label"))
                if true_label_raw is None:
                    continue # Skip if no true label
                try:
                    true_label = int(float(true_label_raw))
                    if true_label in [0, 1]:
                        true_labels_all_records.append(true_label)
                except (ValueError, TypeError):
                    continue # Skip if label not convertible

        if true_labels_all_records:
            true_0s = true_labels_all_records.count(0)
            true_1s = true_labels_all_records.count(1)
            true_total = len(true_labels_all_records)
            true_ratio_str = "N/A"
            if true_0s > 0:
                true_ratio = true_1s / true_0s
                true_ratio_str = f"{true_1s}/{true_0s} ({true_ratio:.2f}) (from {true_total} records with labels)"
            elif true_1s > 0:
                true_ratio_str = f"{true_1s}/{true_0s} (Inf) (from {true_total} records with labels)"
            else:
                true_ratio_str = f"0/0 (N/A) (from {true_total} records with labels)"
            logger.info(f"True Label Distribution for {filename}: Success Ratio (1s/0s) = {true_ratio_str}")
        else:
            logger.warning(f"No valid true labels found in {filename} to calculate distribution.")

    except Exception as e:
        logger.error(f"Error during true label pre-scan for {filepath}: {e}", exc_info=True)


    # Existing logic for processing predictions and metrics follows...
    true_labels_basic, predicted_labels_basic = [], []
    true_labels_pro, predicted_labels_pro = [], []
    true_labels_std, predicted_labels_std = [], [] # Initialize std lists

    processed_records_std = 0
    
    skipped_due_to_error_field = 0
    skipped_due_to_missing_label = 0
    
    for record_idx, record in enumerate(data):
        if not isinstance(record, dict):
            logger.warning(f"Skipping non-dictionary record at index {record_idx}: {record}")
            continue

        if record.get("error") is not None:
            skipped_due_to_error_field += 1
            continue
            
        true_label_raw = record.get("label")
        if true_label_raw is None:
            skipped_due_to_missing_label +=1
            continue
        
        try:
            true_label = int(float(true_label_raw))
            if true_label not in [0, 1]:
                logger.warning(f"Skipping record {record.get('original_index', record_idx)} due to invalid true label value: {true_label_raw}")
                skipped_due_to_missing_label +=1
                continue
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping record {record.get('original_index', record_idx)} due to non-convertible true label '{true_label_raw}': {e}")
            skipped_due_to_missing_label +=1
            continue

        if file_type_hint in ["ssff_nl", "ssff_regular"]:
            pred_basic = get_single_prediction(record, file_type_hint, sub_model_type="Basic")
            if pred_basic is not None:
                true_labels_basic.append(true_label)
                predicted_labels_basic.append(pred_basic)

            pred_pro = get_single_prediction(record, file_type_hint, sub_model_type="Pro")
            if pred_pro is not None:
                true_labels_pro.append(true_label)
                predicted_labels_pro.append(pred_pro)
        
        elif file_type_hint: # For non-SSFF files
            prediction = get_single_prediction(record, file_type_hint)
            if prediction is not None:
                true_labels_std.append(true_label) # Corrected list
                predicted_labels_std.append(prediction) # Corrected list
                processed_records_std += 1
        else:
            logger.warning(f"Skipping record {record_idx} due to unknown file type hint after explicit checks.")


    logger.info(f"Total records in file: {len(data)}")
    logger.info(f"Records skipped due to pre-existing 'error' field: {skipped_due_to_error_field}")
    logger.info(f"Records skipped due to missing/invalid true label: {skipped_due_to_missing_label}")

    if file_type_hint in ["ssff_nl", "ssff_regular"]:
        logger.info(f"SSFF Basic - Records with valid prediction: {len(predicted_labels_basic)}")
        calculate_and_print_metrics(true_labels_basic, predicted_labels_basic, f"{file_type_hint.upper()} - Basic Analysis", filename, all_results_summary)
        
        logger.info(f"SSFF Pro - Records with valid prediction: {len(predicted_labels_pro)}")
        calculate_and_print_metrics(true_labels_pro, predicted_labels_pro, f"{file_type_hint.upper()} - Pro Analysis", filename, all_results_summary)

    elif file_type_hint: # For non-SSFF files
        logger.info(f"{file_type_hint.upper()} - Records with valid prediction: {processed_records_std}")
        calculate_and_print_metrics(true_labels_std, predicted_labels_std, file_type_hint.upper(), filename, all_results_summary)
    else:
        logger.error(f"Could not evaluate {filename} due to unrecognised file type.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results from JSON files.")
    parser.add_argument("--input_files", nargs='*', help="Paths to the JSON result files.", required=True)
    parser.add_argument("--output_csv", help="Optional path to save results to a CSV file.")
    parser.add_argument("--output_plot_dir", help="Optional directory to save comparison plots.")

    args = parser.parse_args()
    all_results_summary = []

    if not args.input_files:
        logger.error("No input files provided. Use --input_files argument.")
        sys.exit(1)

    for filepath in args.input_files:
        filename = os.path.basename(filepath)
        evaluate_results(filepath, all_results_summary)

    summary_df = pd.DataFrame(all_results_summary)

    if args.output_csv:
        try:
            # Ensure the directory for the CSV exists
            csv_output_dir = os.path.dirname(args.output_csv)
            if csv_output_dir: # Check if there is a directory part
                os.makedirs(csv_output_dir, exist_ok=True)
            summary_df.to_csv(args.output_csv, index=False)
            logger.info(f"Saved evaluation summary to {args.output_csv}")
        except Exception as e:
            logger.error(f"Failed to save summary CSV to {args.output_csv}: {e}")
    else:
        print("\n--- Overall Summary Table ---")
        print(summary_df.to_string())

    # Generate and save plots if an output directory for plots is specified
    if args.output_plot_dir:
        plot_key_metrics(summary_df, args.output_plot_dir)
    else:
        logger.info("No output plot directory specified. Skipping plot generation.")

if __name__ == "__main__":
    main() 