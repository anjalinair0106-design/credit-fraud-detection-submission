"""Credit card fraud detection project pipeline , SUBMITTED BY :(20240802167,20240802187,20240802188)"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'backend')))

print("Script starting...")  # Debug print

# Import OS utilities for folder creation.
import os

# Import warnings to suppress non-critical warnings.
import warnings

# Import numpy for numeric operations.
import numpy as np

# Import pandas for data handling.
import pandas as pd

# Import matplotlib for plotting graphs.
import matplotlib.pyplot as plt

# Import seaborn for better data visualizations.
import seaborn as sns

# Import train-test split for model training.
from sklearn.model_selection import train_test_split

# Import preprocessing tools.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import classification models.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import evaluation metrics.
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# Try to import CatBoost if available.
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# Ignore unnecessary warnings.
warnings.filterwarnings("ignore")

# Define random seed for reproducibility.
RANDOM_STATE = 42

# Define dataset file path (use standardized backend dataset for consistency).
DATA_PATH = r"C:\Users\Anjali\OneDrive\Documents\Playground\credit_fraud_project\src\backend\backend\artifacts\fraud_transactions_standardized.csv"

# Define output folders.
BASE_OUTPUT = r"C:\Users\Anjali\OneDrive\Documents\Playground\credit_fraud_project\submission_outputs"
PLOT_DIR = os.path.join(BASE_OUTPUT, "plots")
WAREHOUSE_DIR = os.path.join(BASE_OUTPUT, "warehouse")
RESULT_DIR = os.path.join(BASE_OUTPUT, "results")

# Create output folders if they do not exist.
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# Define helper function for clean console headings.
def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_and_preprocess_data():
    global df
    # Load the fraud dataset.
    df = pd.read_csv(DATA_PATH)

    # Convert transaction date to datetime format.
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    # Remove duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Create time-based features.
    df["hour"] = df["TransactionDate"].dt.hour
    df["day"] = df["TransactionDate"].dt.day
    df["month"] = df["TransactionDate"].dt.month
    df["day_of_week"] = df["TransactionDate"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Create amount-based features.
    df["amount_log"] = np.log1p(df["Amount"])
    df["high_value_transaction"] = (df["Amount"] >= df["Amount"].quantile(0.90)).astype(int)


def generate_plots():
    # Plot fraud vs non-fraud count.
    plt.figure(figsize=(7, 4))
    sns.countplot(x="IsFraud", data=df, palette="Set2")
    plt.title("Fraud vs Non-Fraud Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fraud_count.png"))
    plt.close()

    # Plot amount distribution.
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Amount"], bins=30, kde=True, color="skyblue")
    plt.title("Amount Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "amount_distribution.png"))
    plt.close()

    # Plot fraud by transaction type.
    type_summary = df.groupby(["TransactionType", "IsFraud"]).size().reset_index(name="Count")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=type_summary, x="TransactionType", y="Count", hue="IsFraud", palette="coolwarm")
    plt.title("Fraud by Transaction Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fraud_by_type.png"))
    plt.close()

    # Plot fraud by location.
    location_summary = df.groupby(["Location", "IsFraud"]).size().reset_index(name="Count")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=location_summary, x="Location", y="Count", hue="IsFraud", palette="Paired")
    plt.title("Fraud by Location")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fraud_by_location.png"))
    plt.close()

    # Plot fraud trend over time.
    df["YearMonth"] = df["TransactionDate"].dt.to_period("M").astype(str)
    trend = df.groupby("YearMonth")["IsFraud"].sum().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=trend, x="YearMonth", y="IsFraud", marker="o")
    plt.title("Fraud Trend Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fraud_trend.png"))
    plt.close()


def split_and_engineer_features():
    global X_train, X_test, y_train, y_test, preprocessor
    # Split features and target.
    X = df.drop(columns=["IsFraud", "TransactionID"])
    y = df["IsFraud"]

    # Split data using stratified sampling.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Rebuild train and test tables with target for safe aggregate features.
    train_df = X_train.copy()
    train_df["IsFraud"] = y_train.values
    test_df = X_test.copy()
    test_df["IsFraud"] = y_test.values

    # Create merchant-based aggregate features from training data only.
    merchant_count = train_df.groupby("MerchantID").size().to_dict()
    merchant_fraud_rate = train_df.groupby("MerchantID")["IsFraud"].mean().to_dict()
    location_fraud_rate = train_df.groupby("Location")["IsFraud"].mean().to_dict()

    # Create global fallback fraud rate.
    global_fraud_rate = train_df["IsFraud"].mean()

    # Add merchant and location aggregate features to training data.
    train_df["merchant_transaction_count"] = train_df["MerchantID"].map(merchant_count)
    train_df["merchant_fraud_rate"] = train_df["MerchantID"].map(merchant_fraud_rate)
    train_df["location_fraud_rate"] = train_df["Location"].map(location_fraud_rate)

    # Add merchant and location aggregate features to test data.
    test_df["merchant_transaction_count"] = test_df["MerchantID"].map(merchant_count).fillna(0)
    test_df["merchant_fraud_rate"] = test_df["MerchantID"].map(merchant_fraud_rate).fillna(global_fraud_rate)
    test_df["location_fraud_rate"] = test_df["Location"].map(location_fraud_rate).fillna(global_fraud_rate)

    # Separate final train and test features.
    X_train = train_df.drop(columns=["IsFraud"])
    X_test = test_df.drop(columns=["IsFraud"])
    y_train = train_df["IsFraud"]
    y_test = test_df["IsFraud"]

    # Identify numeric and categorical columns.
    numeric_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Build preprocessing pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


# Define helper function for threshold tuning and metrics.
def evaluate_model(y_true, y_prob):
    thresholds = precision_recall_curve(y_true, y_prob)[2]
    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    y_pred = (y_prob >= best_threshold).astype(int)

    return {
        "threshold": float(best_threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_models():
    global results, random_forest_model, rf_feature_names
    import pickle
    from pathlib import Path
    
    model_path = Path(r"c:\Users\Anjali\OneDrive\Documents\Playground\credit_fraud_project\src\backend\backend\artifacts\fraud_model_bundle.pkl")
    
    if model_path.exists():
        # Load the pre-trained model from backend
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        loaded_model = bundle['model']
        model_name = bundle['model_name']
        backend_preprocessor = bundle['preprocessor']
        backend_engineer = bundle['feature_engineer']
        
        # Use backend's feature engineering and preprocessing for consistency
        X_test_fe = backend_engineer.transform(X_test)
        X_test_processed = backend_preprocessor.transform(X_test_fe)
        
        # Get predictions
        test_prob = loaded_model.predict_proba(X_test_processed)[:, 1]
        test_metrics = evaluate_model(y_test, test_prob)
        
        # Initialize results with loaded model
        results = {model_name: test_metrics}
        
        # If it's RandomForest, set for feature importance
        if hasattr(loaded_model, 'feature_importances_'):
            random_forest_model = loaded_model
            # For feature names, use backend's
            rf_feature_names = backend_preprocessor.get_feature_names_out()
    else:
        # Fallback: Train models as before
        # Build and train Logistic Regression model.
        logistic_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        logistic_model.fit(X_train, y_train)
        logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
        logistic_metrics = evaluate_model(y_test, logistic_prob)


        # Build and train Random Forest model.
        random_forest_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ],
        )
        random_forest_model.fit(X_train, y_train)
        random_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]
        random_forest_metrics = evaluate_model(y_test, random_forest_prob)


        # Initialize results dictionary with sklearn model outputs.
        results = {
            "LogisticRegression": logistic_metrics,
            "RandomForest": random_forest_metrics,
        }


        # Train CatBoost model if the package is installed.
        if CATBOOST_AVAILABLE:
            X_train_cb = X_train.copy()
            X_test_cb = X_test.copy()
            cat_cols = X_train_cb.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X_train_cb.select_dtypes(exclude=["object"]).columns.tolist()

            for col in cat_cols:
                X_train_cb[col] = X_train_cb[col].fillna("Missing").astype(str)
                X_test_cb[col] = X_test_cb[col].fillna("Missing").astype(str)

            for col in num_cols:
                median_value = X_train_cb[col].median()
                X_train_cb[col] = X_train_cb[col].fillna(median_value)
                X_test_cb[col] = X_test_cb[col].fillna(median_value)

            class_weight_1 = (len(y_train) - y_train.sum()) / y_train.sum()

            catboost_model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=0,
                random_seed=RANDOM_STATE,
                class_weights=[1.0, float(class_weight_1)],
            )

            catboost_model.fit(X_train_cb, y_train, cat_features=cat_cols)
            catboost_prob = catboost_model.predict_proba(X_test_cb)[:, 1]
            results["CatBoost"] = evaluate_model(y_test, catboost_prob)
        # Fallback: Train models as before
        # Build and train Logistic Regression model.
        logistic_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        logistic_model.fit(X_train, y_train)
        logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
        logistic_metrics = evaluate_model(y_test, logistic_prob)


        # Build and train Random Forest model.
        random_forest_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ],
        )
        random_forest_model.fit(X_train, y_train)
        random_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]
        random_forest_metrics = evaluate_model(y_test, random_forest_prob)


        # Initialize results dictionary with sklearn model outputs.
        results = {
            "LogisticRegression": logistic_metrics,
            "RandomForest": random_forest_metrics,
        }


        # Train CatBoost model if the package is installed.
        if CATBOOST_AVAILABLE:
            X_train_cb = X_train.copy()
            X_test_cb = X_test.copy()
            cat_cols = X_train_cb.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X_train_cb.select_dtypes(exclude=["object"]).columns.tolist()

            for col in cat_cols:
                X_train_cb[col] = X_train_cb[col].fillna("Missing").astype(str)
                X_test_cb[col] = X_test_cb[col].fillna("Missing").astype(str)

            for col in num_cols:
                median_value = X_train_cb[col].median()
                X_train_cb[col] = X_train_cb[col].fillna(median_value)
                X_test_cb[col] = X_test_cb[col].fillna(median_value)

            class_weight_1 = (len(y_train) - y_train.sum()) / y_train.sum()

            catboost_model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=0,
                random_seed=RANDOM_STATE,
                class_weights=[1.0, float(class_weight_1)],
            )

            catboost_model.fit(X_train_cb, y_train, cat_features=cat_cols)
            catboost_prob = catboost_model.predict_proba(X_test_cb)[:, 1]
            results["CatBoost"] = evaluate_model(y_test, catboost_prob)


def evaluate_and_save_results():
    global results_df, rf_importance
    # Convert model results to a DataFrame.
    results_df = pd.DataFrame(results).T

    # Save model comparison metrics.
    results_df.to_csv(os.path.join(RESULT_DIR, "model_metrics.csv"))

    # Print model comparison in console.
    section("MODEL RESULTS")
    print(results_df.to_string())


    # Extract Random Forest feature importance.
    if hasattr(random_forest_model, 'feature_importances_'):
        # Use rf_feature_names if set (from loaded model), else get from preprocessor
        if 'rf_feature_names' in globals():
            feature_names = rf_feature_names
        elif hasattr(random_forest_model, 'named_steps'):  # It's a Pipeline
            feature_names = random_forest_model.named_steps["preprocessor"].get_feature_names_out()
        else:  # Fallback
            feature_names = [f"feature_{i}" for i in range(len(random_forest_model.feature_importances_))]
        
        rf_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": random_forest_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Save Random Forest feature importance.
        rf_importance.to_csv(os.path.join(RESULT_DIR, "random_forest_feature_importance.csv"), index=False)

        # Plot top Random Forest features.
        plt.figure(figsize=(10, 6))
        sns.barplot(data=rf_importance.head(15), x="importance", y="feature", palette="viridis")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "random_forest_feature_importance.png"))
        plt.close()
    else:
        print("No Random Forest model available for feature importance.")


def build_warehouse():
    # Build star schema tables for ETL and warehouse design.
    df["DateKey"] = df["TransactionDate"].dt.strftime("%Y%m%d").astype(int)

    # Create Dim_Date table.
    dim_date = pd.DataFrame(
        {
            "DateKey": df["DateKey"],
            "FullDate": df["TransactionDate"].dt.date.astype(str),
            "Day": df["TransactionDate"].dt.day,
            "Month": df["TransactionDate"].dt.month,
            "Year": df["TransactionDate"].dt.year,
            "DayOfWeek": df["TransactionDate"].dt.day_name(),
            "WeekendFlag": df["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(int),
        }
    ).drop_duplicates()

    # Create Dim_Merchant table.
    dim_merchant = df.groupby("MerchantID").agg(
        MerchantTransactionCount=("MerchantID", "size"),
        MerchantFraudRate=("IsFraud", "mean"),
    ).reset_index()

    # Create Dim_Location table.
    dim_location = df.groupby("Location").agg(
        LocationFraudRate=("IsFraud", "mean"),
    ).reset_index()

    # Create Dim_TransactionType table.
    dim_transaction_type = df[["TransactionType"]].drop_duplicates().reset_index(drop=True)

    # Create Fact_Transactions table.
    fact_transactions = df[
        ["TransactionID", "DateKey", "MerchantID", "Location", "TransactionType", "Amount", "IsFraud"]
    ].copy()

    # Save warehouse tables.
    dim_date.to_csv(os.path.join(WAREHOUSE_DIR, "Dim_Date.csv"), index=False)
    dim_merchant.to_csv(os.path.join(WAREHOUSE_DIR, "Dim_Merchant.csv"), index=False)
    dim_location.to_csv(os.path.join(WAREHOUSE_DIR, "Dim_Location.csv"), index=False)
    dim_transaction_type.to_csv(os.path.join(WAREHOUSE_DIR, "Dim_TransactionType.csv"), index=False)
    fact_transactions.to_csv(os.path.join(WAREHOUSE_DIR, "Fact_Transactions.csv"), index=False)


    # Create OLAP summaries.
    olap_date = df.groupby("YearMonth")["IsFraud"].sum().reset_index(name="FraudCount")
    olap_merchant = df.groupby("MerchantID")["IsFraud"].sum().reset_index(name="FraudCount")
    olap_location = df.groupby("Location")["IsFraud"].sum().reset_index(name="FraudCount")
    olap_type = df.groupby("TransactionType")["IsFraud"].sum().reset_index(name="FraudCount")

    # Save OLAP summaries.
    olap_date.to_csv(os.path.join(WAREHOUSE_DIR, "OLAP_Fraud_By_Date.csv"), index=False)
    olap_merchant.to_csv(os.path.join(WAREHOUSE_DIR, "OLAP_Fraud_By_Merchant.csv"), index=False)
    olap_location.to_csv(os.path.join(WAREHOUSE_DIR, "OLAP_Fraud_By_Location.csv"), index=False)
    olap_type.to_csv(os.path.join(WAREHOUSE_DIR, "OLAP_Fraud_By_TransactionType.csv"), index=False)


def save_summary():
    # Save a text summary file for report writing.
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as file:
        file.write("Credit Card Fraud Detection Project Summary\n")
        file.write("=" * 50 + "\n")
        file.write(f"Dataset shape: {df.shape}\n")
        file.write(f"Class distribution: {df['IsFraud'].value_counts().to_dict()}\n\n")
        if 'results_df' in globals():
            file.write("Model Results:\n")
            file.write(results_df.to_string())
            file.write("\n")
        else:
            file.write("Model results not available (training skipped).\n")


    # Print final folder locations.
    section("OUTPUT SAVED")
    print(f"Plots saved in: {PLOT_DIR}")
    print(f"Warehouse tables saved in: {WAREHOUSE_DIR}")
    print(f"Results saved in: {RESULT_DIR}")


# Main execution
if __name__ == "__main__":
    load_and_preprocess_data()
    generate_plots()
    split_and_engineer_features()
    train_models()
    evaluate_and_save_results()
    build_warehouse()
    save_summary()
