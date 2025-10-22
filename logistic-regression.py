import os
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc


# Files used to persist the trained model and test set for reuse
MODEL_FILE = "logistic_model.joblib"
METADATA_FILE = "logistic_model_metadata.json"
TESTSET_FILE = "logistic_testset.joblib"


def load_data(csv_path="creditcard.csv"):
    data = pd.read_csv(csv_path)
    return data


def evaluate_and_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)

    print("AUC Score:", auc_score)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def main(retrain=False, csv_path="creditcard.csv"):
    data = load_data(csv_path)

    print(data.head())
    print("\n\nRows and columns: ", data.shape)
    print("\n\nColun names: ", data.columns)
    print("\n\n", data["Class"].value_counts())
    print("Statistical summary: \n\n", data.describe())

    # Amount spent distribution and boxplot
    data["Amount"].hist(bins=50)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.show()

    sns.boxplot(x="Class", y="Amount", data=data)
    plt.title("Transaction Amounts by Class")
    plt.show()

    # Prepare features and labels
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # If a saved model exists and retrain is not requested, load it and evaluate on the saved test set
    if os.path.exists(MODEL_FILE) and os.path.exists(TESTSET_FILE) and not retrain:
        print(f"Loading saved model from {MODEL_FILE} and test set from {TESTSET_FILE}...")
        model = joblib.load(MODEL_FILE)
        test_set = joblib.load(TESTSET_FILE)
        X_test = test_set["X_test"]
        y_test = test_set["y_test"]

        # Sanity check: ensure columns match (if you later change features this will warn)
        saved_columns = None
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                meta = json.load(f)
                saved_columns = meta.get("feature_columns")

        if saved_columns and list(X.columns) != saved_columns:
            print("Warning: current CSV features do not match the features used to train the saved model.")
            print("Saved columns:", saved_columns)
            print("Current columns:", list(X.columns))
            print("Proceeding to evaluate, but consider retraining with --retrain if the dataset changed.")

        evaluate_and_plot(model, X_test, y_test)
        return

    # Otherwise we train a new model, save it and a consistent test set for future runs
    print("Training a new model (this will be saved for future runs)...")

    # NOTE: the original script used SMOTE on the whole dataset before splitting.
    # We'll follow the same approach to keep behavior identical, but in a production setup you'd
    # normally split first then apply SMOTE only on the training partition.
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("Balanced shape after SMOTE:", X_res.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model and test set for reuse
    joblib.dump(model, MODEL_FILE)
    joblib.dump({"X_test": X_test, "y_test": y_test}, TESTSET_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump({"feature_columns": list(X.columns)}, f)

    print(f"Saved trained model to {MODEL_FILE}")
    print(f"Saved test set to {TESTSET_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")

    evaluate_and_plot(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or load a logistic regression model for fraud detection."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of the model even if a saved model exists.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="creditcard.csv",
        help="Path to the CSV file (default: creditcard.csv).",
    )
    args = parser.parse_args()
    main(retrain=args.retrain, csv_path=args.csv)
