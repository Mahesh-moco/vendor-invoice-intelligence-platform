from data_preprocessing import (
    load_invoice_data,
    apply_label,
    split_data,
    scale_features
)

from modeling_evaluation import (
    train_knn_classifier,
    evaluate_classifier
)

import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

TARGET = "Flag_invoice"


def main():

    # Load data
    df = load_invoice_data()

    # Apply invoice risk labels
    df = apply_label(df)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test
    )

    # Train the KNN classifier
    knn_model = train_knn_classifier(
        X_train_scaled,
        y_train
    )

    # Evaluate the model
    evaluate_classifier(
        knn_model,
        X_test_scaled,
        y_test,
        "KNN Classifier"
    )

    # Save the trained model
    joblib.dump(
        knn_model,
        "models/predict_flag_invoice.pkl"
    )

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()