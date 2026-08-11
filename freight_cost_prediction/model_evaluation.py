 #Import machine learning models
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Import model evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model

from sklearn.neighbors import KNeighborsRegressor

def train_knn_regressor(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors (KNN) Regressor model.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a regression model using MAE, RMSE, and R² score.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"\nModel: {model_name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")
    
    # Return metrics
    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }
























