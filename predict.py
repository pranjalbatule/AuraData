import pandas as pd
import joblib

def make_predictions():
    print("[Prediction Script] Loading saved model and new data...")
    
    # 1. Load the trained model from disk
    model_filename = "best_trained_model.pkl"
    try:
        model = joblib.load(model_filename)
        print(f"Successfully loaded model from '{model_filename}'")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Run agent_system.py first.")
        return

    # 2. Incoming raw data (must include 'first_name' just like the training data)
    new_data = pd.DataFrame({
        'customer_id': [5, 6],
        'first_name': ['Bob', 'Diana'],
        'sales_amount': [320.00, 1500.00]
    })
    
    print("\n--- New Raw Data Input ---")
    print(new_data)
    
    # 3. Preprocess new data identically to training data
    # Standardize headers
    new_data.columns = [str(col).strip().lower().replace(" ", "_") for col in new_data.columns]
    
    # One-hot encode categorical features
    new_data_encoded = pd.get_dummies(new_data, columns=['first_name'], drop_first=True)
    
    # Align columns: ensure every feature the model was trained on is present in the new data
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in new_data_encoded.columns:
                new_data_encoded[col] = 0 # Fill missing dummy categories with 0
        # Reorder columns to match the exact training order
        new_data_encoded = new_data_encoded[model.feature_names_in_]

    # 4. Make Predictions using the aligned data
    predictions = model.predict(new_data_encoded)
    
    # 5. Attach predictions and display
    new_data['predicted_outcome'] = predictions
    
    print("\n--- Prediction Results ---")
    print(new_data)
    print("--------------------------")

if __name__ == "__main__":
    make_predictions()