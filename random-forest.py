import os
import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from joblib import dump, load
import numpy as np

MODEL_PATH = "rf_model.joblib"


def load_data(path="creditcard.csv"):
    data = pd.read_csv(path)
    return data


def visualize(data):
    # Amount spent distribution
    plt.figure(figsize=(10, 6))
    data['Amount'].hist(bins=50)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.show()

    # Boxplot for Amount by Class (fraud/not fraud)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=data)
    plt.title("Transaction Amounts by Class")
    plt.show()

    # Class distribution visualization
    plt.figure(figsize=(8, 6))
    data['Class'].value_counts().plot(kind='bar')
    plt.title("Class Distribution (0: Normal, 1: Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def prepare_data(data):
    X = data.drop('Class', axis=1)  # Features
    y = data['Class']  # Labels
    return X, y


def resample_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def train_and_save_model(X_train, y_train, X_columns, model_path=MODEL_PATH):
    model = RandomForestClassifier(
        n_estimators=100,          # Number of trees
        max_depth=10,              # Maximum depth of trees
        min_samples_split=5,       # Minimum samples required to split
        min_samples_leaf=2,        # Minimum samples required at leaf node
        random_state=42,
        n_jobs=-1                  # Use all available cores
    )

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Save model and columns together so we can verify ordering on load
    dump({'model': model, 'columns': list(X_columns)}, model_path)
    print(f"Model saved to {model_path}")
    return model


def load_model_if_exists(X_columns, model_path=MODEL_PATH, force_retrain=False):
    """
    Load a previously saved model if the file exists and the feature columns match.
    Returns (model, loaded_flag). If force_retrain is True, always returns (None, False)
    so the caller will retrain.
    """
    if force_retrain:
        print("Force retrain requested; skipping model load.")
        return None, False

    if not os.path.exists(model_path):
        print("No existing model found.")
        return None, False

    try:
        saved = load(model_path)
        saved_model = saved.get('model')
        saved_columns = saved.get('columns')
        if saved_model is None or saved_columns is None:
            warnings.warn("Saved model file does not contain expected structure. Will retrain.")
            return None, False

        # Check that the saved columns match current columns (ignoring order)
        if set(saved_columns) != set(X_columns):
            warnings.warn(
                "Feature columns differ between saved model and current data. "
                "Saved columns will be compared; if they match as a set the columns will be reordered. "
                "If they don't match, retraining will happen."
            )
            # If same set but different order, reordering will be handled later by caller.
            if set(saved_columns) == set(X_columns):
                print("Saved model columns are the same set as current columns. Will reorder features to match saved model.")
                return saved_model, True  # indicate loaded, but caller must reorder X
            else:
                print("Saved model columns do not match current columns. Will retrain.")
                return None, False
        else:
            print(f"Loaded model from {model_path} and columns match.")
            return saved_model, True

    except Exception as e:
        warnings.warn(f"Failed to load existing model due to error: {e}. Will retrain.")
        return None, False


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n=== Model Performance ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Detailed classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)

    print(f"\nAUC Score: {auc_score:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Feature Importance (only if model has the attribute)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== Top 10 Most Important Features ===")
        print(feature_importance.head(10))

        # Plot Feature Importance
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # Model parameters summary
    if hasattr(model, "n_estimators"):
        print("\n=== Model Parameters ===")
        print(f"Number of estimators: {model.n_estimators}")
        print(f"Max depth: {model.max_depth}")
        print(f"Min samples split: {model.min_samples_split}")
        print(f"Min samples leaf: {model.min_samples_leaf}")


def main(args):
    data = load_data(args.data_path)
    if args.no_visuals is False:
        visualize(data)

    X, y = prepare_data(data)

    # Try loading existing model
    model, loaded_flag = load_model_if_exists(X.columns, model_path=MODEL_PATH, force_retrain=args.retrain)

    if loaded_flag and model is not None:
        # If model loaded but saved columns order may differ, reorder test columns later.
        # Still need to train-test split for evaluation: we will resample and split
        X_res, y_res = resample_smote(X, y)
        print("Original Shape:", X.shape)
        print("Balanced Shape:", X_res.shape)
        print("Original Class Distribution:\n", pd.Series(y).value_counts())
        print("Balanced Class Distribution:\n", pd.Series(y_res).value_counts())

        # If loaded model expects a specific column order, reorder X_res to match
        saved = load(MODEL_PATH)
        saved_columns = saved.get('columns', list(X.columns))
        if list(saved_columns) != list(X_res.columns):
            if set(saved_columns) == set(X_res.columns):
                X_res = X_res[saved_columns]
                print("Reordered features to match saved model's column order.")
            else:
                # This case should not happen because load_model_if_exists would have forced retrain,
                # but keep a safe check.
                warnings.warn("Saved model columns do not match current data; retraining to be safe.")
                model = None
                loaded_flag = False

    if not loaded_flag or model is None:
        # Need to train
        X_res, y_res = resample_smote(X, y)
        print("Original Shape:", X.shape)
        print("Balanced Shape:", X_res.shape)
        print("Original Class Distribution:\n", pd.Series(y).value_counts())
        print("Balanced Class Distribution:\n", pd.Series(y_res).value_counts())

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        model = train_and_save_model(X_train, y_train, X_train.columns, model_path=MODEL_PATH)
    else:
        # We already computed X_res and y_res above; split and evaluate
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Evaluate regardless of whether model was loaded or just trained
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest fraud detection (with model persistence).")
    parser.add_argument("--data-path", type=str, default="creditcard.csv", help="Path to the CSV dataset.")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model even if a saved model exists.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip plotting visualizations (useful for CI or headless runs).")
    args = parser.parse_args()
    main(args)
