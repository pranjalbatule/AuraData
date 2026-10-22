import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error

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

# 3. Data Engineer Node
def data_engineer_node(state: DataAgentState) -> Dict[str, Any]:
    print(f"\n[Data Engineer Agent] Performing deep dataset audit & cleaning...")
    df = state["df"]
    
    if df is None:
        return {"current_step": "end", "execution_logs": ["Error: No dataframe found for data engineer."]}
    
    plan = []
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        plan.append(f"Dropped {duplicate_count} exact duplicate rows.")
    else:
        plan.append("No duplicate rows found.")
        
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

    print(f"[Data Engineer Agent] Cleaning Plan Completed.")
    return {
        "df": df,
        "cleaning_plan": plan,
        "execution_logs": state.get("execution_logs", []) + plan,
        "current_step": "feature_engineering"  
    }

# 4. Feature Engineering & Transformation Node
def feature_engineering_node(state: DataAgentState) -> Dict[str, Any]:
    print(f"\n[Feature Engineering Agent] Transforming features for data science readiness...")
    df = state["df"]
    logs = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'customer_id' in numeric_cols:
        numeric_cols.remove('customer_id') 
        
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logs.append(f"Standardized numerical columns: {numeric_cols}")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logs.append(f"One-hot encoded categorical columns: {cat_cols}")

    print(f"[Feature Engineering Agent] Transformation complete.")
    return {
        "df": df,
        "execution_logs": state.get("execution_logs", []) + logs,
        "current_step": "visualization"
    }

# 5. Intelligent Visualization Node (Dynamically picks charts based on data types)
def intelligent_visualization_node(state: DataAgentState) -> Dict[str, Any]:
    print(f"\n[Visualization Agent] Inspecting data types and generating targeted charts...")
    df = state["df"]
    logs = []
    
    os.makedirs("generated_charts", exist_ok=True)
    
    # Generate Heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Intelligent Correlation Heatmap")
        path_heatmap = "generated_charts/correlation_heatmap.png"
        plt.savefig(path_heatmap, bbox_inches='tight')
        plt.close()
        logs.append(f"Generated Correlation Heatmap: {path_heatmap}")

    # Dynamically chart individual columns
    for col in df.columns:
        if col == 'customer_id':
            continue 
            
        plt.figure(figsize=(6, 4))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True, color="teal")
            plt.title(f"Distribution & Density: {col}")
            chart_path = f"generated_charts/dist_{col}.png"
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            logs.append(f"Generated Distribution Plot for '{col}': {chart_path}")
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            top_cats = df[col].value_counts().head(10)
            sns.barplot(x=top_cats.values, y=top_cats.index, palette="viridis", hue=top_cats.index, legend=False)
            plt.title(f"Frequency Count: {col}")
            chart_path = f"generated_charts/count_{col}.png"
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            logs.append(f"Generated Frequency Bar Plot for '{col}': {chart_path}")

    print(f"[Visualization Agent] Targeted charts generated successfully.")
    return {
        "execution_logs": state.get("execution_logs", []) + logs,
        "current_step": "model_training"
    }

# 6. Advanced Model Tournament Node
def model_tournament_node(state: DataAgentState) -> Dict[str, Any]:
    print(f"\n[Model Tournament Agent] Benchmarking multiple algorithms...")
    df = state["df"]
    logs = []
    
    target_col = df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    is_classification = y.nunique() < 20 and not pd.api.types.is_numeric_dtype(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model = None
    best_score = float('-inf') if is_classification else float('inf')
    best_model_name = ""
    
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            logs.append(f"Model '{name}' Accuracy: {score * 100:.2f}%")
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = mean_squared_error(y_test, preds)
            logs.append(f"Model '{name}' MSE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_model = model
                best_model_name = name

    logs.append(f"🏆 Winner Selected: '{best_model_name}' with score {best_score:.4f}")
    
    model_filename = "best_trained_model.pkl"
    joblib.dump(best_model, model_filename)
    logs.append(f"Saved winning model to disk as '{model_filename}'")
    print(f"[Model Tournament Agent] Winning model saved as '{model_filename}'!")

    return {
        "execution_logs": state.get("execution_logs", []) + logs,
        "current_step": "completed"
    }

# 7. Complete Test Runner Block
if __name__ == "__main__":
    dummy_csv_path = "messy_sample.csv"
    with open(dummy_csv_path, "w") as f:
        f.write("Customer ID,First Name,Sales Amount\n")
        f.write("1,Alice,250.50\n")
        f.write("2,Bob,\n")
        f.write("3,Charlie,450.00\n")
        f.write("4,Diana,300.00\n")
        f.write("1,Alice,250.50\n")
    
    state = universal_ingestion_node({
        "raw_file_path": dummy_csv_path, "file_type": "", "df": None,
        "cleaning_plan": [], "eda_insights": [], "execution_logs": [], "final_report": "", "current_step": "ingestion"
    })
    
    if state["current_step"] == "data_engineering":
        state = data_engineer_node(state)
        
    if state["current_step"] == "feature_engineering":
        state = feature_engineering_node(state)
        
    if state["current_step"] == "visualization":
        state = intelligent_visualization_node(state)
        output_file = "cleaned_output_data.csv"
        state["df"].to_csv(output_file, index=False)
        
    if state["current_step"] == "model_training":
        state = model_tournament_node(state)
        
    print(f"\n[Pipeline Complete] Full multi-agent workflow executed successfully!")


    from langgraph.graph import StateGraph, START, END

# 1. Initialize the StateGraph using our DataAgentState schema
workflow = StateGraph(DataAgentState)

# 2. Add each agent function as a node in the graph
workflow.add_node("ingestion_agent", universal_ingestion_node)
workflow.add_node("data_engineer_agent", data_engineer_node)
workflow.add_node("feature_engineer_agent", feature_engineering_node)
workflow.add_node("visualization_agent", intelligent_visualization_node)
workflow.add_node("model_tournament_agent", model_tournament_node)

# 3. Define the execution flow (edges) connecting the nodes sequentially
workflow.add_edge(START, "ingestion_agent")
workflow.add_edge("ingestion_agent", "data_engineer_agent")
workflow.add_edge("data_engineer_agent", "feature_engineer_agent")
workflow.add_edge("feature_engineer_agent", "visualization_agent")
workflow.add_edge("visualization_agent", "model_tournament_agent")
workflow.add_edge("model_tournament_agent", END)

# 4. Compile the graph into an executable application
aura_graph = workflow.compile()

# 5. Running the LangGraph App
if __name__ == "__main__":
    dummy_csv_path = "messy_sample.csv"
    with open(dummy_csv_path, "w") as f:
        f.write("Customer ID,First Name,Sales Amount\n")
        f.write("1,Alice,250.50\n")
        f.write("2,Bob,\n")
        f.write("3,Charlie,450.00\n")
        
    # Kick off the graph with initial state
    initial_state = {
        "raw_file_path": dummy_csv_path,
        "file_type": "",
        "df": None,
        "cleaning_plan": [],
        "eda_insights": [],
        "execution_logs": [],
        "final_report": "",
        "current_step": "ingestion"
    }
    
    print("\n🚀 [LangGraph] Invoking multi-agent autonomous workflow graph...")
    final_output_state = aura_graph.invoke(initial_state)
    print("\n✅ [LangGraph] Workflow completed successfully through state management!")