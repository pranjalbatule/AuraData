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

# 2. Build the Universal Ingestion Router Node
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
            
        # Standardize column names (snake_case conversion)
        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
        
        print(f"[Ingestion Agent] Success! Converted into DataFrame.")
        print(f"--- DataFrame Preview ---")
        print(df.head())
        print(f"-------------------------")
        
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

# 3. Test Runner Block
if __name__ == "__main__":
    # Let's create a quick messy dummy CSV file to test our ingestion agent right now!
    dummy_csv_path = "messy_sample.csv"
    with open(dummy_csv_path, "w") as f:
        f.write("Customer ID,First Name, Sales Amount \n")
        f.write("1, Alice, 250.50\n")
        f.write("2, Bob, 120.00\n")
    
    # Initialize our Agent State pointing to this file
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
    
    # Run our ingestion node!
    result_state = universal_ingestion_node(initial_state)
    print("\n[System Test Complete] Next step route assigned to:", result_state.get("current_step"))