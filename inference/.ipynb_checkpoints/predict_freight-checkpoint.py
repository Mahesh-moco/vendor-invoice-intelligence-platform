import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


# Load the trained freight prediction model.
def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


# Predict freight cost for new vendor invoices.
def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame
        Input data with predicted freight cost.
    """
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Freight"] = model.predict(input_df).round(2)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "Dollars": [185000,9000,3000,200]
    }

    predictions = predict_freight_cost(sample_data)

    print(predictions)