import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
import json
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Importing the functions
from number_of_biased_features import (
    feature_bias_pipeline, 
    comprehensive_bias_visualization, 
    visualize_multigroup_fairness,
    load_exclusion_config,
    preprocess_sensitive_attributes,
    auto_detect_categorical_columns,
    should_exclude_column
)

SENSITIVE_CONFIG_PATH = "sensitive_attributes.json"
EXCLUSION_CONFIG_PATH = "columns_to_exclude.json"

def load_columns(file_obj):
    """Load CSV and return list of columns for dropdown."""
    try:
        df = pd.read_csv(file_obj.name)
        cols = df.columns.tolist()
        # Update dropdown with choices and pick first column as default (optional)
        return gr.Dropdown(choices=cols, value=cols[0] if cols else None), ""
    except Exception as e:
        return gr.Dropdown(choices=[], value=None), f"Failed to read CSV file: {str(e)}"

def run_bias_analysis_with_json(file_obj, target_col):
    if not file_obj:
        return "Please upload a dataset.", None
    if not target_col:
        return "Please select a target column.", None

    try:
        # Load dataset
        df = pd.read_csv(file_obj.name)
        target_col = target_col.strip()

        if target_col not in df.columns:
            return f"Error: Target column '{target_col}' not found.", None
        
        # Load configurations
        try:
            exclusion_config = load_exclusion_config(EXCLUSION_CONFIG_PATH)
            with open(SENSITIVE_CONFIG_PATH, "r") as f:
                sensitive_config = json.load(f)
        except Exception as e:
            return f"Failed to load sensitive attributes config JSON: {str(e)}", None
    
        # Filter sensitive attributes present in the dataset
        available_sensitive_attrs = []
        filtered_sensitive_config = {"sensitive_attributes": []}
        for attr_config in sensitive_config.get("sensitive_attributes", []):
            attr_name = attr_config["name"]
            if attr_name in df.columns:
                filtered_sensitive_config["sensitive_attributes"].append(attr_config)
                available_sensitive_attrs.append(attr_name)

        if not available_sensitive_attrs:
            return "No sensitive attribute from JSON found in uploaded dataset.", None
        
        # Preprocess sensitive attributes using your existing function
        df_processed, encoding_info = preprocess_sensitive_attributes(df, filtered_sensitive_config)
        print(f"Preprocessed sensitive attributes: {list(encoding_info.keys())}")
    
        # Prepare input X and target y
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # Convert target to numeric if needed (handle 'Yes'/'No')
        original_target_type = y.dtype
        if y.dtype == object:
            unique_vals = set(y.dropna().unique())
            if unique_vals <= {'Yes', 'No'}:
                y = y.map({'Yes': 1, 'No': 0}).fillna(y)
            elif unique_vals <= {'True', 'False'}:
                y = y.map({'True': 1, 'False': 0})
            elif unique_vals <= {True, False}:
                y = y.astype(int)
            elif len(unique_vals) == 2:
                # Binary categorical - use label encoding
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            else:
                return "Error: Target column must be numeric or binary('Yes'/'No').", None
            
        # Auto-detect and encode categorical columns (excluding sensitive attributes and target)
        columns_to_check = [col for col in X.columns 
                           if col not in available_sensitive_attrs]
        columns_to_encode = [col for col in columns_to_check 
                           if not should_exclude_column(col, exclusion_config)[0]]
        categorical_cols = auto_detect_categorical_columns(X[columns_to_encode])
        
        for col in categorical_cols:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
    
        # Run the bias detection pipeline for each sensitive attribute found and collect summaries
        summaries = []
        images = []
        temp_dir = tempfile.mkdtemp()
        overall_bias_detected = False

        # Set matplotlib backend for web interface
        plt.switch_backend('Agg')
        plt.ioff()  # Turn off interactive mode

        for sensitive_attr in available_sensitive_attrs:
            try:
                print(f"\nAnalyzing bias for: {sensitive_attr}")
                
                # Run bias pipeline with your existing function
                results = feature_bias_pipeline(
                    X, y, 
                    sensitive_attr=sensitive_attr, 
                    target=target_col,
                    exclusion_config=exclusion_config,
                    apply_mitigation=True
                )
                # Add after calling feature_bias_pipeline
                print(f"Keys in results dictionary: {list(results.keys())}")
                print(f"'mitigation_results' in results: {'mitigation_results' in results}")
                if 'mitigation_results' in results:
                    print(f"Mitigation results keys: {list(results['mitigation_results'].keys())}")

                # Extract key metrics
                biased_count = results.get('final_bias_count', 0)
                total_features = len(results.get('important_features', []))
                direct_bias = results.get('direct_bias_results', {}).get('has_direct_bias', False)
                model_bias = results.get('model_results', {}).get('bias_detected', False)
                proxy_count = len(results.get('proxy_candidates', []))
                
                # Get direct bias p-value for better assessment
                direct_bias_pvalue = results.get('direct_bias_results', {}).get('p_value', 1.0)
                
                # **CORRECTED BIAS DETECTION LOGIC**
                # Bias is detected if ANY of these conditions are met:
                # 1. Significant biased features found (> 0)
                # 2. Model bias detected
                # 3. Significant proxy features found (> 0)
                # 4. Strong direct bias (p-value < 0.001 for very strong evidence)
                
                significant_direct_bias = direct_bias and direct_bias_pvalue < 0.001
                has_bias = (biased_count > 0) or model_bias or (proxy_count > 0) or significant_direct_bias
                
                if has_bias:
                    overall_bias_detected = True
                
                # Create status with better logic
                if has_bias:
                    if biased_count > 0 or proxy_count > 0:
                        bias_status = "BIAS DETECTED"
                    elif significant_direct_bias:
                        bias_status = "IRECT BIAS ONLY"
                    else:
                        bias_status = "MODEL BIAS ONLY"
                else:
                    bias_status = "NO SIGNIFICANT BIAS"
                
                summary = f"{sensitive_attr}: {bias_status}\n"
                summary += f"   • Biased Features: {biased_count}/{total_features}\n"
                summary += f"   • Direct Bias: {'YES' if direct_bias else 'NO'}"
                
                # Add p-value for direct bias if significant
                if direct_bias and direct_bias_pvalue is not None:
                    if direct_bias_pvalue < 0.001:
                        summary += f" (p < 0.001)"
                    elif direct_bias_pvalue < 0.01:
                        summary += f" (p < 0.01)"
                    elif direct_bias_pvalue < 0.05:
                        summary += f" (p < 0.05)"
                    else:
                        summary += f" (p = {direct_bias_pvalue:.3f})"
                summary += "\n"
                
                summary += f"   • Model Bias: {'YES' if model_bias else 'NO'}\n"
                summary += f"   • Proxy Features: {proxy_count}\n"
                
                # Add top biased features if any
                if results.get('biased_features') and biased_count > 0:
                    top_biased = [feat for feat, _ in results['biased_features'][:3]]
                    summary += f"   • Top Biased: {', '.join(top_biased)}\n"
                
                # Add explanation for edge cases
                if direct_bias and biased_count == 0 and proxy_count == 0 and not model_bias:
                    if significant_direct_bias:
                        summary += f"   • Note: Strong direct correlation but no feature-level bias\n"
                    else:
                        summary += f"   • Note: Weak direct bias, likely not actionable\n"

                # --- NEW: Add mitigation result summary ---
                if 'mitigation_results' in results and results['mitigation_results']:
                    summary += "\n" + "-" * 30 + "\n"  # Add a separator line
                    summary += " • Mitigation Results:\n"
                    baseline_sp = results['model_results'].get('detailed_fairness_metrics', {}).get('statistical_parity', 0)
                    
                    summary += " • Mitigation Results:\n"
                    summary += f"Baseline Statistical Parity (before mitigation): {baseline_sp:.4f}\n"

                    best_method = None
                    best_sp = baseline_sp

                    # Process pre-processing methods
                    if 'preprocessing' in results['mitigation_results']:
                        for method_name, metrics in results['mitigation_results']['preprocessing'].items():
                            sp = metrics.get('statistical_parity', float('inf'))
                            summary += f"    - Preprocessing ({method_name}): SP = {sp:.4f}"
                            
                            # Calculate reduction percentage
                            if baseline_sp > 0 and sp < baseline_sp:
                                reduction = ((baseline_sp - sp) / baseline_sp) * 100
                                summary += f" ({reduction:.1f}% reduction)"
                            
                            summary += "\n"
                            
                            # Track best method
                            if sp < best_sp:
                                best_sp = sp
                                best_method = f"Preprocessing ({method_name})"
    
                    # Process in-processing methods
                    if 'inprocessing' in results['mitigation_results']:
                        for method_name, metrics in results['mitigation_results']['inprocessing'].items():
                            sp = metrics.get('statistical_parity', float('inf'))
                            summary += f"    - In-processing ({method_name}): SP = {sp:.4f}"
                            
                            # Calculate reduction percentage
                            if baseline_sp > 0 and sp < baseline_sp:
                                reduction = ((baseline_sp - sp) / baseline_sp) * 100
                                summary += f" ({reduction:.1f}% reduction)"
                            
                            summary += "\n"
                            
                            # Track best method
                            if sp < best_sp:
                                best_sp = sp
                                best_method = f"In-processing ({method_name})"
                    
                    # Process post-processing methods
                    if 'postprocessing' in results['mitigation_results']:
                        for method_name, metrics in results['mitigation_results']['postprocessing'].items():
                            sp = metrics.get('statistical_parity', float('inf'))
                            summary += f"    - Post-processing ({method_name}): SP = {sp:.4f}"
                            
                            # Calculate reduction percentage
                            if baseline_sp > 0 and sp < baseline_sp:
                                reduction = ((baseline_sp - sp) / baseline_sp) * 100
                                summary += f" ({reduction:.1f}% reduction)"
                            
                            summary += "\n"
            
                            # Track best method
                            if sp < best_sp:
                                best_sp = sp
                                best_method = f"Post-processing ({method_name})"

                    # Highlight best method
                    if best_method and best_sp < baseline_sp:
                        improvement = ((baseline_sp - best_sp) / baseline_sp) * 100
                        summary += f"\n    • Best mitigation: {best_method}\n"
                        summary += f"    • Bias reduction: {improvement:.1f}%\n"
                        
                        # Add assessment
                        if improvement > 50:
                            summary += f"    • Assessment: Substantial bias reduction\n"
                        elif improvement > 25:
                            summary += f"    • Assessment: Moderate bias reduction\n"
                        elif improvement > 10:
                            summary += f"    • Assessment: Slight bias reduction\n"
                        else:
                            summary += f"    • Assessment: Minimal bias reduction\n"
                    elif best_sp >= baseline_sp:
                        summary += f"\n    • No effective bias reduction achieved\n"
                
                summaries.append(summary)

                # Generate visualizations
                try:
                    plt.close('all')
                    comprehensive_bias_visualization(X, y, sensitive_attr, results)
                    plot_path = os.path.join(temp_dir, f"bias_analysis_{sensitive_attr}.png")
                    plt.savefig(plot_path, bbox_inches='tight', dpi=100, facecolor='white')
                    plt.close('all')
                    
                    if os.path.exists(plot_path):
                        images.append(plot_path)
                        
                except Exception as viz_error:
                    print(f"Visualization error for {sensitive_attr}: {viz_error}")
                    plt.close('all')

                # Multi-group visualization if needed
                if X[sensitive_attr].nunique() > 2:
                    try:
                        plt.close('all')
                        visualize_multigroup_fairness(X, y, sensitive_attr, results)
                        plot_path_2 = os.path.join(temp_dir, f"multi_group_{sensitive_attr}.png")
                        plt.savefig(plot_path_2, bbox_inches='tight', dpi=100, facecolor='white')
                        plt.close('all')
                        
                        if os.path.exists(plot_path_2):
                            images.append(plot_path_2)
                            
                    except Exception:
                        plt.close('all')

            except Exception as analysis_error:
                summaries.append(f"{sensitive_attr}: Analysis failed - {str(analysis_error)}")

        # Create final summary
        dataset_info = f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns | Target: {target_col}\n"
        dataset_info += f"Analyzed: {', '.join(available_sensitive_attrs)}\n\n"
        
        final_summary = dataset_info + "\n".join(summaries)
        
        # Add overall conclusion with corrected logic
        conclusion = f"\n{'='*50}\n"
        if overall_bias_detected:
            conclusion += f"OVERALL RESULT: ACTIONABLE BIAS FOUND IN DATASET\n"
        else:
            conclusion += f"OVERALL RESULT: NO SIGNIFICANT BIAS DETECTED\n"
        conclusion += f"Generated {len(images)} visualization(s)"
        
        final_summary += conclusion

        return final_summary, images if images else None

    except Exception as e:
        return f"Analysis failed: {str(e)}", None

with gr.Blocks(title="Bias Detection Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Algorithmic Bias Detection Tool')
    gr.Markdown("Upload your dataset and analyze potential bias in Machine Learning approaches")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Dataset (CSV)", 
                file_types=[".csv"],
                type="filepath"
            )
            
            target_dropdown = gr.Dropdown(
                label="Select Target Column", 
                choices=[],
                interactive=True,
                info="Choose the column you want to predict (dependent variable)"
            )
            
            analyze_btn = gr.Button(
                "Run Bias Analysis", 
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            **Requirements:**
            - CSV file with headers
            - Binary target column (0/1, Yes/No, True/False)
            - Sensitive attributes configured in `sensitive_attributes.json`
            """)

    with gr.Row():
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Bias Analysis Results", 
                lines=25,
                max_lines=40,
                show_copy_button=True
            )

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(
                label="Bias Visualizations",
                columns=1,
                rows=2,
                height=600,
                show_share_button=False
            )

    # Event handlers
    file_input.change(
        fn=load_columns, 
        inputs=file_input, 
        outputs=[target_dropdown, output_text]
    )
    
    analyze_btn.click(
        fn=run_bias_analysis_with_json,
        inputs=[file_input, target_dropdown],
        outputs=[output_text, output_gallery]
    )
    
    # Also trigger analysis when target is selected (optional)
    # target_dropdown.change(
    #     fn=run_bias_analysis_with_json,
    #     inputs=[file_input, target_dropdown],
    #     outputs=[output_text, output_gallery]
    # )

if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True if you want to create a public link
        server_name="127.0.0.1",  # Use "0.0.0.0" to allow access from other devices
        server_port=7860,
        show_error=True,
        # show_tips=True
    )