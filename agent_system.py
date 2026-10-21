import os
import json
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional

# 1. Define the Shared Memory Structure
class DataAgentState(TypedDict):
    raw_file_path: str         
    file_type: str             
    df: Optional[Any]          
    cleaning_plan: List[str]   
    eda_insights: List[str]    
    execution_logs: List[str]  
    final_report: str          
    current_step: str          

# 2. Universal Ingestion Router Node
def universal_ingestion_node(state: DataAgentState) -> Dict[str, Any]:
    file_path = state["raw_file_path"]
    ext = os.path.splitext(file_path)[1].lower()
    
    print(f"\n[Ingestion Agent] Processing incoming file: {file_path} (Format: {ext})")
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        elif ext == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
        
        print(f"[Ingestion Agent] Success! Converted into DataFrame.")
        
        return {
            "df": df,
            "file_type": ext.replace(".", ""),
            "execution_logs": state.get("execution_logs", []) + [f"Successfully ingested {file_path} into tabular format."],
            "current_step": "data_engineering" 
        }
        
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        print(f"[Ingestion Agent Error] {error_msg}")
        return {
            "execution_logs": state.get("execution_logs", []) + [error_msg],
            "current_step": "end"
        }

# 3. Data Engineer Node with Imputation, Duplicates, and Outlier Detection
def data_engineer_node(state: DataAgentState) -> Dict[str, Any]:
    print(f"\n[Data Engineer Agent] Performing deep dataset audit & cleaning...")
    df = state["df"]
    
    if df is None:
        return {"current_step": "end", "execution_logs": ["Error: No dataframe found for data engineer."]}
    
    plan = []
    
    # 1. Handle Duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        plan.append(f"Dropped {duplicate_count} exact duplicate rows.")
    else:
        plan.append("No duplicate rows found.")
        
    # 2. Handle Missing Values (Impute or Drop)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df)
            if missing_pct > 0.5:
                df = df.drop(columns=[col])
                plan.append(f"Dropped column '{col}' (>50% missing data).")
            elif pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                plan.append(f"Imputed numerical column '{col}' missing values with median: {median_val}")
            else:
                df[col] = df[col].fillna("Unknown")
                plan.append(f"Imputed text column '{col}' missing values with 'Unknown'.")

    # 3. Detect Outliers in Numerical Columns (using IQR)
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            plan.append(f"Detected {len(outliers)} statistical outliers in column '{col}'. Flagged for review.")

    print(f"[Data Engineer Agent] Cleaning & Audit Plan Completed:")
    for step in plan:
        print(f"  - {step}")
        
    return {
        "df": df,
        "cleaning_plan": plan,
        "execution_logs": state.get("execution_logs", []) + plan,
        "current_step": "completed"
    }

# 4. Test Runner Block with Messy Data
if __name__ == "__main__":
    # Create a test CSV containing missing data, an outlier, and a duplicate row
    dummy_csv_path = "messy_sample.csv"
    with open(dummy_csv_path, "w") as f:
        f.write("Customer ID,First Name,Sales Amount\n")
        f.write("1,Alice,250.50\n")
        f.write("2,Bob,\n")           # Missing value for sales amount
        f.write("3,Charlie,150.00\n")
        f.write("4,Diana,50000.00\n")  # Outlier (extreme high sales amount)
        f.write("1,Alice,250.50\n")   # Exact duplicate of row 1
    
    # Step 1: Run Ingestion
    initial_state: DataAgentState = {
        "raw_file_path": dummy_csv_path,
        "file_type": "",
        "df": None,
        "cleaning_plan": [],
        "eda_insights": [],
        "execution_logs": [],
        "final_report": "",
        "current_step": "ingestion"
    }
    
    state_after_ingestion = universal_ingestion_node(initial_state)
    
    # Step 2: Pass state into Data Engineer if route matches
    if state_after_ingestion["current_step"] == "data_engineering":
        final_state = data_engineer_node(state_after_ingestion)
        print(f"\n[Pipeline Complete] Final cleaned dataframe shape: {final_state['df'].shape}")
        print("\n--- Cleaned DataFrame Preview ---")
        print(final_state["df"])
        print("---------------------------------")