import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\chour\Desktop\numpy\ML_project_2.0\invoice_flagging\models\predict_flag_invoice.pkl"
SCALER_PATH = r"C:\Users\chour\Desktop\numpy\ML_project_2.0\invoice_flagging\models\scaler.pkl"


# Load the trained invoice flag prediction model.
def load_model(model_path=MODEL_PATH):

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


# Load the saved StandardScaler.
def load_scaler(scaler_path=SCALER_PATH):

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    return scaler


# Predict whether an invoice should be flagged.
def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    # Convert input to DataFrame
    input_df = pd.DataFrame(input_data)

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    # Make predictions
    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
    "invoice_quantity": [20],
    "invoice_dollars": [12000],
    "Freight": [50],
    "total_item_quantity": [20],
    "total_item_dollars": [1200],
    "avg_receiving_delay": [2]
}
    predictions = predict_invoice_flag(sample_data)

    print(predictions)