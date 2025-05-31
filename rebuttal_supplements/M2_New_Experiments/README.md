# M2 New Experiments: Baseline Startup Analysis

This directory contains scripts and data for running baseline experiments as part of the Startup Success Forecasting Framework (SSFF) rebuttal and further development. The primary goal is to establish baseline performance using simpler models before evaluating more complex methodologies.

## `run_experimentation_baseline.py`

This script is designed to analyze startups using a basic "vanilla" Large Language Model (LLM) approach. It serves as the first of several comparison methods.

### Core Functionality:

*   **Input**: Takes a CSV file where each row represents a startup, and one column contains the textual description/information about that startup.
*   **Processing**: For each startup, it queries a specified LLM (e.g., GPT-4o-mini, GPT-4o) with a standardized prompt instructing it to act as a venture capitalist. The LLM provides:
    1.  A comprehensive analysis (covering market, product/technology, and founder/team).
    2.  An overall score (1-10).
    3.  A recommendation ("Successful" or "Unsuccessful").
*   **Output**: Results are saved in a JSONL (JSON Lines) file, where each line is a JSON object representing the analysis for one startup. This format is convenient for further processing and analysis.
*   **Customization**: The script supports command-line arguments to specify:
    *   The LLM model to use.
    *   The path to the input CSV dataset.
    *   The name of the column containing the startup information.
    *   The number of rows to process from the dataset.
    *   The path for the output JSONL file.

### Purpose:

The `run_experimentation_baseline.py` script helps establish a performance baseline using a straightforward LLM setup. This allows for a clear comparison against more sophisticated methods planned for evaluation, such as:

*   Chain-of-Thought (CoT) prompting.
*   The full SSFF framework (with and without its machine learning components).
*   Other established frameworks like FoundersGPT or methodologies from relevant research papers (e.g., RAISE).

### Sample Usage Commands:

All commands should be run from the `rebuttal_supplements/M2_New_Experiments/` directory.

1.  **Run with default settings:**
    *   Uses `gpt-4o-mini`.
    *   Processes `data/curated_dataset_success_10pct.csv`.
    *   Assumes startup info is in a column named `description`.
    *   Saves output to `results_baseline_gpt-4o-mini_curated_dataset_success_10pct.jsonl`.

    ```bash
    python run_experimentation_baseline.py
    ```

2.  **Specify a different dataset and model, and process only the first 50 rows:**
    *   Uses `gpt-4o`.
    *   Processes `data/curated_dataset_success_20pct.csv`.
    *   Saves output to `results_gpt4o_20pct_first50.jsonl`.

    ```bash
    python run_experimentation_baseline.py \
        --dataset_path data/curated_dataset_success_20pct.csv \
        --model_name gpt-4o \
        --num_rows 50 \
        --output_file results_gpt4o_20pct_first50.jsonl
    ```

3.  **Specify the column containing startup information if it's not 'description':**
    *   Uses `o1-mini`.
    *   Processes `data/my_custom_dataset.csv`.
    *   Assumes startup info is in a column named `startup_text_data`.

    ```bash
    python run_experimentation_baseline.py \
        --dataset_path data/my_custom_dataset.csv \
        --model_name o1-mini \
        --info_column startup_text_data \
        --output_file results_o1mini_custom_data.jsonl
    ```

4.  **Process all rows of a specific dataset with a specific model:**

    ```bash
    python run_experimentation_baseline.py \
        --dataset_path data/curated_dataset_success_50pct.csv \
        --model_name gpt-4o
    ```
    *(Output file will be auto-generated, e.g., `results_baseline_gpt-4o_curated_dataset_success_50pct.jsonl`)*

### Next Steps:

After running baseline experiments, the results from this script can be compared against those obtained from other analysis methods (CoT, full SSFF, etc.) to quantify the benefits of more advanced techniques. 