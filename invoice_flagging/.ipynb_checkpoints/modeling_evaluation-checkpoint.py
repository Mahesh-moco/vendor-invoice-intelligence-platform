from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Train a K-Nearest Neighbors (KNN) classifier model.
def train_knn_classifier(X_train, y_train):

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    model.fit(X_train, y_train)

    return model

# Evaluate a classification model using common performance metrics.
def evaluate_classifier(model, X_test, y_test, model_name):

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print(f"\nModel: {model_name}")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Return results
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }