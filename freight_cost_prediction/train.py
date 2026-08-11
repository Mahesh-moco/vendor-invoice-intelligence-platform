from pathlib import Path
import joblib

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_knn_regressor,
    evaluate_model
)


def main():
    # Database path
    db_path = r"C:\Users\chour\Downloads\inventory_ml_project1.db"

    # Create models directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare features and target
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    knn_model = train_knn_regressor(X_train, y_train)

    # Evaluate models
    results = []
    results.append(
        evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    )
    results.append(
        evaluate_model(knn_model, X_test, y_test, "KNeighborsRegressor")
    )

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["MAE"])
    best_model_name = best_model_info["Model"]

    # Dictionary of trained models
    models = {
        "Linear Regression": lr_model,
        "KNeighborsRegressor": knn_model
    }

    best_model = models[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()