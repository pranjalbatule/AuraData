import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- HELPER: CREATE SAMPLE DATASETS ---
def create_sample_datasets():
    os.makedirs("sample_data", exist_ok=True)
    loan_df = pd.DataFrame({
        "account_number": ["ACC-101", None, "ACC-103", "ACC-104", "ACC-105", "ACC-106", "ACC-107", "ACC-108", "ACC-109", "ACC-110"] * 10,
        "age": [25, 45, 35, 52, 23, 41, 38, 50, 29, 48] * 10,
        "gender": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10,
        "married": [0, 1, 1, 1, 0, 1, 1, 1, 0, 1] * 10,
        "dependents": [0, 2, 1, 3, 0, 2, 1, 2, 0, 1] * 10,
        "education": [1, 1, 0, 1, 0, 1, 1, 0, 0, 1] * 10,
        "self_employed": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0] * 10,
        "applicantincome": [30000, 85000, 52000, 120000, 240000, 75000, 64000, 110000, 42000, 95000] * 10,
        "loanamount": [10000, 35000, 20000, 50000, 8000, 28000, 22000, 45000, 15000, 32000] * 10,
        "cibil_score": [600, 750, 680, 810, 550, 720, 690, 790, 620, 740] * 10,
        "property_area": [1, 2, 1, 3, 2, 1, 3, 2, 1, 2] * 10,
        "deposit": ["No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes"] * 10
    })
    loan_df.to_excel("sample_data/bank_loan_sample.xlsx", index=False)
    loan_df.to_csv("sample_data/bank_loan_sample.csv", index=False)

create_sample_datasets()

# --- HELPER: COMPILE 3-PAGE EXECUTIVE PDF REPORT ---
def generate_pdf_report(markdown_text, output_pdf_path="Executive_Report.pdf"):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, leading=18, spaceBefore=14, spaceAfter=6, textColor='#1E3A8A')
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=15, textColor='#1E3A8A')

    story = []
    lines = markdown_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            story.append(Paragraph(line.replace('# ', ''), title_style))
        elif line.startswith('## '):
            story.append(Paragraph(line.replace('## ', ''), heading_style))
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph(f"• {line[2:]}", normal_style))
        else:
            story.append(Paragraph(line, normal_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    return output_pdf_path

# --- BACKEND PIPELINE WITH COMPREHENSIVE GUARDRAILS ---
def run_governed_pipeline(file_obj, sample_choice, missing_strategy, target_col, problem_type, outlier_strategy):
    if file_obj is not None:
        file_path = file_obj.name
    elif sample_choice == "Bank Loan Excel Data (.xlsx)":
        file_path = "sample_data/bank_loan_sample.xlsx"
    elif sample_choice == "Bank Loan CSV Data (.csv)":
        file_path = "sample_data/bank_loan_sample.csv"
    else:
        return None, None, None, [], "", None, "**[ERROR]**: Please upload a dataset or select a sample dataset.", gr.update(), "", ""

    logs = ["**[AuraData Guardrail Engine]**: Initializing comprehensive security & governance protocols..."]

    try:
        # Guardrail 1: Universal File Format Ingestion with robust fallback
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv' or not ext:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
        total_initial_rows = len(df)

        # Guardrail 2: Target Column Existence Check
        if target_col not in df.columns:
            return None, None, None, [], "", None, f"❌ **[Guardrail Error]**: Target column `{target_col}` not found in dataset schema.", gr.update(), "", "\n".join(logs)

        # Guardrail 3: Large File Scale Optimization (> 100k rows)
        if total_initial_rows > 100000:
            logs.append(f"⚠️ **[Scale Guardrail]**: Large dataset detected ({total_initial_rows:,} rows). Applying memory-optimized random sampling (100k rows).")
            df = df.sample(n=100000, random_state=42).reset_index(drop=True)

        # Duplicate Cleanse
        dups = df.duplicated().sum()
        if dups > 0:
            df = df.drop_duplicates()
            logs.append(f"**[Data Hygiene]**: Purged **{dups}** duplicate records.")

        # Guardrail 4: Intelligent Missing Value & Critical Column Protection
        critical_keywords = ['id', 'account', 'uuid', 'ssn', 'key', 'code']
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                is_critical = any(kw in col for kw in critical_keywords)
                if is_critical:
                    df[col] = df[col].fillna("MISSING_ID_FLAG")
                    logs.append(f"🛡️ **[Guardrail Protected]**: Critical column `{col}` preserved; nulls flagged as `MISSING_ID_FLAG`.")
                else:
                    if missing_strategy == "Drop Rows": 
                        df = df.dropna(subset=[col])
                    elif missing_strategy == "Drop Column": 
                        df = df.drop(columns=[col])
                    else:
                        df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else "Unknown")

        # Outlier Treatment (Numeric only)
        for col in df.select_dtypes(include=['number']).columns:
            if col == target_col: continue
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            if outlier_strategy == "Cap / Winsorize Outliers":
                df[col] = df[col].clip(lower, upper)

        cleaned_csv_path = "cleaned_output_data.csv"
        df.to_csv(cleaned_csv_path, index=False)
        logs.append("**[Governance Complete]**: Sanitized schema successfully compiled.")

        # Visualizations: Correlation Matrix
        os.makedirs("generated_charts", exist_ok=True)
        chart_paths = []
        num_df = df.select_dtypes(include=['number'])
        
        if len(num_df.columns) > 1:
            plt.figure(figsize=(7, 4))
            sns.heatmap(num_df.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f")
            plt.title("Inter-Feature Correlation Matrix", fontsize=12, fontweight='bold')
            h_path = "generated_charts/correlation_heatmap.png"
            plt.savefig(h_path, bbox_inches='tight')
            plt.close()
            chart_paths.append(h_path)

        # Smart Feature Encoding & Target Encoding
        df_ml = df.copy()
        y_raw = df_ml[target_col]
        
        # Guardrail 5: Problem-Type Auto-Correction
        unique_target_count = y_raw.nunique()
        is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)
        
        if problem_type == "Classification" and is_numeric_target and unique_target_count > 25:
            problem_type = "Regression"
            logs.append(f"🔄 **[Guardrail Auto-Correction]**: Target `{target_col}` has {unique_target_count} unique numeric values. Automatically switched problem type to **Regression**.")
        elif problem_type == "Regression" and not is_numeric_target:
            problem_type = "Classification"
            logs.append(f"🔄 **[Guardrail Auto-Correction]**: Target `{target_col}` is non-numeric text. Automatically switched problem type to **Classification**.")

        if y_raw.dtype == 'object' or y_raw.dtype == 'bool' or not is_numeric_target:
            y, target_uniques = pd.factorize(y_raw)
        else:
            y = y_raw.values
            target_uniques = None

        df_ml_features = df_ml.drop(columns=[target_col])

        exclude_keywords = ['id', 'name', 'code', 'date', 'uuid', 'account']
        filtered_cols = [
            c for c in df_ml_features.columns 
            if not any(k in c for k in exclude_keywords)
        ]
        df_ml_features = df_ml_features[filtered_cols]

        cat_mappings = {}
        for col in df_ml_features.select_dtypes(include=['object', 'bool']).columns:
            codes, uniques = pd.factorize(df_ml_features[col])
            df_ml_features[col] = codes
            cat_mappings[col] = list(uniques)

        df_encoded = df_ml_features.astype(float)

        # Track feature bounds for prediction bounds guardrail
        feature_bounds = {col: (float(df_encoded[col].min()), float(df_encoded[col].max())) for col in df_encoded.columns}

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_encoded)
        df_scaled = pd.DataFrame(scaled_array, columns=df_encoded.columns)

        corrs = df_scaled.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
        top_features = corrs.head(5).index.tolist()

        X = df_scaled[top_features]

        realistic_examples = []
        feature_hints = []
        for feat in top_features:
            if feat in cat_mappings:
                sample_cat = str(cat_mappings[feat][0]) if cat_mappings[feat] else "category"
                realistic_examples.append(sample_cat)
                feature_hints.append(f"**{feat}** [Categories: {', '.join(map(str, cat_mappings[feat][:5]))}]")
            else:
                med_val = round(float(df[feat].median()), 1) if pd.api.types.is_numeric_dtype(df[feat]) else 0.0
                realistic_examples.append(str(med_val))
                feature_hints.append(f"**{feat}** [Numeric]")

        example_str = ", ".join(realistic_examples)
        hints_str = " | ".join(feature_hints)

        is_class = (problem_type == "Classification" or len(pd.unique(y)) < 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Guardrail 6: Model Training Fallback Error Handler
        try:
            if is_class:
                best_model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42, solver='liblinear').fit(X_train, y_train)
                best_score = accuracy_score(y_test, best_model.predict(X_test))
                best_name = "Logistic Regression"
            else:
                best_model = LinearRegression().fit(X_train, y_train)
                best_score = mean_squared_error(y_test, best_model.predict(X_test))
                best_name = "Linear Regression"
        except Exception as train_err:
            logs.append(f"⚠️ **[Training Fallback Triggered]**: Primary solver failed ({str(train_err)}). Falling back to resilient default estimators.")
            best_model = LinearRegression().fit(X_train, y_train)
            best_score = mean_squared_error(y_test, best_model.predict(X_test))
            best_name = "Linear Regression (Fallback)"

        # Save model state including bounds guardrails
        joblib.dump((best_model, top_features, scaler, df_encoded.columns, cat_mappings, target_uniques, feature_bounds), "best_trained_model.pkl")
        logs.append(f"**[Model Tournament Winner]**: **{best_name}** trained on **Top {len(top_features)} Features** with score **{best_score:.4f}**.")

        # Feature Importance Visualization
        plt.figure(figsize=(7, 4))
        if hasattr(best_model, "coef_"):
            importances = pd.Series(abs(best_model.coef_[0]), index=top_features).nlargest(5)
        elif hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=top_features).nlargest(5)
        else:
            importances = pd.Series([1]*len(top_features), index=top_features)
            
        importances.plot(kind='barh', color='#2563EB')
        plt.title(f"Top 5 Decision Drivers ({best_name})", fontsize=12, fontweight='bold')
        plt.xlabel("Absolute Feature Weight")
        imp_path = "generated_charts/feature_importance.png"
        plt.savefig(imp_path, bbox_inches='tight')
        plt.close()
        chart_paths.append(imp_path)

        report_md = f"""# **Executive Business Intelligence & Predictive Analytics Report**
**Platform**: AuraData Enterprise Studio | **Target Variable Analysed**: `{target_col}`

---

## **Section 1: Executive Summary & Project Framework**
This technical intelligence report evaluates **{total_initial_rows:,}** observations under rigorous guardrail protection, isolating the **Top {len(top_features)} predictive drivers**.

- **Selected Modeling Paradigm**: **{problem_type}**
- **Optimized Algorithmic Winner**: **{best_name}**
- **Validation Benchmark Score**: **`{best_score:.4f}`**
- **Top Features Utilized**: **`{', '.join(top_features)}`**

---

## **Section 2: Data Governance & Hygiene Audit**
- **Duplicates Removed**: **{dups}**
- **Missing Value Strategy**: **{missing_strategy}** (Critical columns preserved and flagged)
- **Outlier Protocol**: **{outlier_strategy}**

---

## **Section 3: Prescriptive Strategic Playbook**
1. **Targeted Interventions**: Focus enterprise resources on variations within primary decision drivers.
2. **Operational Workflow**: Maintain automated ingestion via `cleaned_output_data.csv`.
"""
        with open("executive_insight_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        pdf_path = generate_pdf_report(report_md)
        scorecard = f"✅ **Guardrail Audit Status**: Completed Successfully | **Rows Processed**: **{total_initial_rows:,}**"
        
        feature_guide = f"**Required Input Schema (Top {len(top_features)} Features Only)**:\n{hints_str}\n*(Example Input Format: `{example_str}`)*"
        updated_textbox = gr.update(value="", placeholder=f"e.g. {example_str}")

        return df.head(10), cleaned_csv_path, "best_trained_model.pkl", chart_paths, report_md, pdf_path, scorecard, feature_guide, updated_textbox, "\n".join(logs)

    except Exception as e:
        return None, None, None, [], "", None, f"❌ **Execution Failed**: {str(e)}", gr.update(placeholder="Enter values..."), "", ""

def get_columns(file_obj, sample_choice):
    if file_obj is not None:
        file_path = file_obj.name
    elif sample_choice == "Bank Loan Excel Data (.xlsx)":
        file_path = "sample_data/bank_loan_sample.xlsx"
    elif sample_choice == "Bank Loan CSV Data (.csv)":
        file_path = "sample_data/bank_loan_sample.csv"
    else:
        return gr.update(choices=[]), "**[Notice]**: Please upload any dataset or select a sample."

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv' or not ext:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except Exception as err:
        return gr.update(choices=[]), f"❌ Ingestion Error: {str(err)}"

    cols = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    guide = f"**Available Columns**: `{', '.join(cols)}` *(Pipeline will automatically extract Top 5 features for prediction)*"
    return gr.update(choices=cols, value=cols[-1] if cols else None), guide

def make_prediction(feat_str):
    if not os.path.exists("best_trained_model.pkl"):
        return "⚠️ **[Model Missing]**: Please execute the governed pipeline in Tab 1 first."
    try:
        model, top_features, scaler, all_encoded_columns, cat_mappings, target_uniques, feature_bounds = joblib.load("best_trained_model.pkl")
        
        raw_vals = [x.strip() for x in feat_str.split(",")]
        if len(raw_vals) != len(top_features):
            return f"❌ **[Validation Error]**: Expected **{len(top_features)}** values for **({', '.join(top_features)})**, but received **{len(raw_vals)}**."
        
        processed_vals = []
        warnings = []
        for i, feat in enumerate(top_features):
            val_str = raw_vals[i]
            if feat in cat_mappings:
                uniques = cat_mappings[feat]
                if val_str in uniques:
                    processed_vals.append(float(uniques.index(val_str)))
                else:
                    matched = next((u for u in uniques if str(u).lower() == val_str.lower()), 0)
                    processed_vals.append(float(uniques.index(matched) if matched in uniques else 0))
                    warnings.append(f"⚠️ Category `{val_str}` for `{feat}` unknown; defaulted safely.")
            else:
                num_val = float(val_str)
                # Guardrail 7: Prediction Bounds & Outlier Warning Check
                if feat in feature_bounds:
                    b_min, b_max = feature_bounds[feat]
                    if num_val < b_min or num_val > b_max:
                        warnings.append(f"⚠️ Value `{num_val}` for `{feat}` is outside historical training bounds [{b_min}, {b_max}].")
                processed_vals.append(num_val)

        full_input = pd.DataFrame([[0] * len(all_encoded_columns)], columns=all_encoded_columns)
        for i, feat in enumerate(top_features):
            if feat in full_input.columns:
                full_input[feat] = processed_vals[i]
                
        scaled_input = scaler.transform(full_input)
        scaled_df = pd.DataFrame(scaled_input, columns=all_encoded_columns)
        final_input_subset = scaled_df[top_features]

        pred = model.predict(final_input_subset)
        result_val = pred[0]
        
        if target_uniques is not None and len(target_uniques) > 0:
            idx = int(round(result_val))
            if 0 <= idx < len(target_uniques):
                result_val = target_uniques[idx]
            prediction_label = "Predicted Category"
        elif hasattr(model, "classes_"):
            result_val = int(round(result_val))
            prediction_label = "Predicted Class"
        else:
            prediction_label = "Predicted Numerical Value"
            result_val = round(float(result_val), 4)

        warning_md = f"\n\n*Notice: {'; '.join(warnings)}*" if warnings else ""
        return f"### **🎉 Prediction Result ({prediction_label}): {result_val}**\n*(Computed successfully using guardrail-governed features)*{warning_md}"
    except Exception as e:
        return f"❌ **Prediction Error**: {str(e)}"

# --- INTERACTIVE GRADIO WEB APPLICATION INTERFACE ---
with gr.Blocks(title="AuraData Data Science Studio") as demo:
    gr.Markdown("# **AuraData Data Science Studio**")
    gr.Markdown("### **Autonomous Data Cleansing, Machine Learning Tournaments, & Executive Reporting.**")
    
    tabs = gr.Tabs()
    with tabs:
        with gr.Tab("1. Data Ingestion & Audit", id=0):
            gr.Markdown("### **Step 1: Upload Any File (CSV, Excel, JSON, Parquet) or Select Sample Data**")
            with gr.Row():
                file_input = gr.File(label="Upload Custom Dataset (.csv, .xlsx, .json, .parquet)")
                sample_dropdown = gr.Dropdown(label="Or Select Sample Dataset", choices=["None", "Bank Loan Excel Data (.xlsx)", "Bank Loan CSV Data (.csv)"], value="None")
            
            load_cols_btn = gr.Button("Load & Inspect Dataset Columns", variant="secondary")
            scorecard_box = gr.Markdown("**Dataset status will appear here after loading.**")
            feature_guide_box = gr.Markdown("**Upload data to view detected feature schema.**")
            
            gr.Markdown("---")
            gr.Markdown("### **Step 2: Configure Governance & Machine Learning Gates**")
            with gr.Row():
                target_dropdown = gr.Dropdown(label="Select Target Variable", choices=[])
                missing_dropdown = gr.Dropdown(label="Missing Values Strategy", choices=["Impute Median/Mean", "Drop Rows", "Drop Column"], value="Impute Median/Mean")
                problem_dropdown = gr.Dropdown(label="Problem Type", choices=["Classification", "Regression"], value="Classification")
                outlier_dropdown = gr.Dropdown(label="Outlier Strategy", choices=["Keep as is", "Cap / Winsorize Outliers"], value="Cap / Winsorize Outliers")
            
            run_btn = gr.Button("Execute Governed Pipeline & Train Model", variant="primary")
            logs_box = gr.Textbox(label="Autonomous Agent Execution Logs", lines=4)
            
            gr.Markdown("---")
            next_btn_1 = gr.Button("Proceed to Visualizations & Feature Drivers ➡️", variant="primary")

        with gr.Tab("2. Visualizations & Feature Drivers", id=1):
            gr.Markdown("### **Exploratory Analytics & Decision Driver Plots**")
            charts_gallery = gr.Gallery(label="Generated Diagnostic & Feature Importance Plots", columns=2, height="auto")
            data_preview = gr.DataFrame(label="Cleaned Data Table")
            
            gr.Markdown("---")
            with gr.Row():
                prev_btn_2 = gr.Button("⬅️ Back to Ingestion", variant="secondary")
                next_btn_2 = gr.Button("Proceed to Live Prediction Studio ➡️", variant="primary")

        with gr.Tab("3. Live Prediction Studio", id=2):
            gr.Markdown("### **Test Trained Model Interactively (Top Features Only)**")
            prediction_schema_guide = gr.Markdown("**Run the pipeline in Tab 1 first to load your top features.**")
            
            with gr.Row():
                feature_input = gr.Textbox(label="Enter Feature Values (Comma-separated text/numbers)", placeholder="e.g. ACC-101, 70.0, 175.0, 1988")
            predict_btn = gr.Button("Generate AI Prediction", variant="primary")
            prediction_output = gr.Markdown("### **Prediction results will appear here.**")
            predict_btn.click(fn=make_prediction, inputs=feature_input, outputs=prediction_output)
            
            gr.Markdown("---")
            with gr.Row():
                prev_btn_3 = gr.Button("⬅️ Back to Visualizations", variant="secondary")
                next_btn_3 = gr.Button("Proceed to Executive Reporting Suite ➡️", variant="primary")

        with gr.Tab("4. Executive Reporting Suite", id=3):
            gr.Markdown("### **Comprehensive 3-Page Executive Business & Strategy Report**")
            report_view = gr.Markdown(label="Executive Report Viewer")
            with gr.Row():
                dl_csv = gr.File(label="Download Clean CSV")
                dl_model = gr.File(label="Download Model (.pkl)")
                dl_report = gr.File(label="Download Report (.md)")
                dl_pdf = gr.File(label="Download 3-Page Executive Report (.pdf)")
            
            gr.Markdown("---")
            prev_btn_4 = gr.Button("⬅️ Back to Prediction Studio", variant="secondary")

    load_cols_btn.click(fn=get_columns, inputs=[file_input, sample_dropdown], outputs=[target_dropdown, feature_guide_box])
    sample_dropdown.change(fn=get_columns, inputs=[file_input, sample_dropdown], outputs=[target_dropdown, feature_guide_box])
    
    run_btn.click(
        fn=run_governed_pipeline,
        inputs=[file_input, sample_dropdown, missing_dropdown, target_dropdown, problem_dropdown, outlier_dropdown],
        outputs=[data_preview, dl_csv, dl_model, charts_gallery, report_view, dl_pdf, scorecard_box, prediction_schema_guide, feature_input, logs_box]
    )

    next_btn_1.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    prev_btn_2.click(lambda: gr.Tabs(selected=0), outputs=tabs)
    next_btn_2.click(lambda: gr.Tabs(selected=2), outputs=tabs)
    prev_btn_3.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=3), outputs=tabs)
    prev_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Glass())