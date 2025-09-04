# Fairness in Machine Learning: Bias Detection Analysis

This project provides a comprehensive toolkit for detecting and analyzing bias in machine learning datasets and models. It helps data scientists and developers identify potential sources of unfairness related to sensitive attributes like gender, race, or age.

## Features

*   **Direct Bias Analysis**: Analyzes the direct statistical relationship between a sensitive attribute and the target variable.
*   **Feature-Level Bias Detection**: Identifies features that may be acting as proxies for sensitive attributes using statistical tests and mutual information.
*   **Model-Based Bias Testing**: Trains a model and evaluates its performance and fairness using a suite of standard fairness metrics.
*   **Comprehensive Fairness Metrics**: Implements various fairness metrics, including:
    *   Statistical Parity
    *   Disparate Impact
    *   Equal Opportunity
    *   Equalized Odds
    *   Calibration
*   **Configuration-Driven Analysis**: Allows users to specify sensitive attributes and columns to exclude from the analysis via JSON configuration files (`sensitive_attributes.json`, `columns_to_exclude.json`).
*   **Rich Visualizations**: Generates a variety of plots to help visualize bias, including target distribution by sensitive group, feature importance, and distribution plots for biased features.

## Getting Started

### Prerequisites

*   Python 3.7+
*   The libraries listed in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

The main entry point for the bias detection analysis is `bias_detection_ui.py`.

1.  **Configure your analysis**:
    *   Edit `sensitive_attributes.json` to define the sensitive columns in your dataset.
    *   Edit `columns_to_exclude.json` to specify any columns that should be ignored during the analysis.

2.  **Run the UI**:
    ```bash
    python bias_detection_ui.py
    ```
    This will launch a user interface that guides you through the process of selecting a dataset and running the bias analysis.

## Project Structure

```
├── number_of_biased_features.py  # Core logic for bias detection and fairness metrics.
├── number_of_biased_features_ui.py # Another UI component.
├── datasets/                     # Sample datasets for analysis.
├── README.md                       # This file.
```

## How it Works

The pipeline performs the following steps:

1.  **Loads Data**: Reads a CSV dataset.
2.  **Loads Configuration**: Reads `sensitive_attributes.json` and `columns_to_exclude.json`.
3.  **Direct Bias Analysis**: Checks for a direct correlation between the sensitive attribute and the target variable.
4.  **Feature Analysis**: Identifies features that are highly correlated with the sensitive attribute (proxy features).
5.  **Model-Based Testing**: Trains a simple model (e.g., Logistic Regression) and calculates fairness metrics based on its predictions.
6.  **Generates Results**: Outputs a summary of biased features, proxy candidates, and fairness metric scores, along with visualizations.
