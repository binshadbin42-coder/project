# ============================================================
# train_model.py
# Stock Market Prediction Project
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "stock_news_11_columns_6000rows.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TRAIN_RATIO = 0.80


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def create_features(file_path):

    print("\nLoading dataset...")

    df = pd.read_csv(file_path)

    required_columns = [
        "Date",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Sentiment_Score",
        "Volatility",
        "Target"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    # Sort by stock and date

    df = df.sort_values(
        ["Symbol", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Sentiment_Score",
        "Volatility",
        "Target"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    fill_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Sentiment_Score",
        "Volatility"
    ]

    df[fill_cols] = (
        df.groupby("Symbol")[fill_cols]
        .transform(
            lambda x: x.ffill().bfill()
        )
    )

    # ========================================================
    # RETURN FEATURES
    # ========================================================

    df["Return_1D"] = (
        df.groupby("Symbol")["Close"]
        .pct_change(1)
    )

    df["Return_3D"] = (
        df.groupby("Symbol")["Close"]
        .pct_change(3)
    )

    df["Return_5D"] = (
        df.groupby("Symbol")["Close"]
        .pct_change(5)
    )

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    df["SMA_5"] = (
        df.groupby("Symbol")["Close"]
        .transform(
            lambda x: x.rolling(5).mean()
        )
    )

    df["SMA_20"] = (
        df.groupby("Symbol")["Close"]
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )

    df["Price_to_SMA20"] = (
        df["Close"] /
        df["SMA_20"]
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    df["Volatility_5D"] = (
        df.groupby("Symbol")["Return_1D"]
        .transform(
            lambda x: x.rolling(5).std()
        )
    )

    # ========================================================
    # LAG FEATURES
    # ========================================================

    for lag in [1, 2, 3]:

        df[f"Close_Lag_{lag}"] = (
            df.groupby("Symbol")["Close"]
            .shift(lag)
        )

        df[f"Sentiment_Lag_{lag}"] = (
            df.groupby("Symbol")["Sentiment_Score"]
            .shift(lag)
        )

    # ========================================================
    # SENTIMENT MOVING AVERAGE
    # ========================================================

    df["Sentiment_SMA_3D"] = (
        df.groupby("Symbol")["Sentiment_Score"]
        .transform(
            lambda x: x.rolling(3).mean()
        )
    )

    # ========================================================
    # TARGETS
    # ========================================================

    # Regression target:
    # Next trading day's closing price

    df["Next_Close"] = (
        df.groupby("Symbol")["Close"]
        .shift(-1)
    )

    # Classification:
    # 0 = DOWN / SELL
    # 1 = UP / BUY

    df["Target"] = df["Target"].astype(int)

    # Remove rows created by rolling/lag calculations

    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# 2. PREPARE DATA
# ============================================================

def prepare_data(df):

    features = [

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        "Sentiment_Score",
        "Volatility",

        "Return_1D",
        "Return_3D",
        "Return_5D",

        "SMA_5",
        "SMA_20",
        "Price_to_SMA20",

        "Volatility_5D",

        "Close_Lag_1",
        "Close_Lag_2",
        "Close_Lag_3",

        "Sentiment_Lag_1",
        "Sentiment_Lag_2",
        "Sentiment_Lag_3",

        "Sentiment_SMA_3D"
    ]

    X = df[features]

    y_reg = df["Next_Close"]

    y_clf = df["Target"]

    # --------------------------------------------------------
    # Chronological 80/20 split
    # --------------------------------------------------------

    split_idx = int(
        len(df) * TRAIN_RATIO
    )

    X_train = X.iloc[:split_idx]

    X_test = X.iloc[split_idx:]

    y_reg_train = y_reg.iloc[:split_idx]

    y_reg_test = y_reg.iloc[split_idx:]

    y_clf_train = y_clf.iloc[:split_idx]

    y_clf_test = y_clf.iloc[split_idx:]

    return (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
        features
    )


# ============================================================
# 3. REGRESSION MODELS
# ============================================================

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("=" * 60)
    print("REGRESSION MODEL TRAINING")
    print("=" * 60)

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Random Forest Regressor":
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting (Default)":
            GradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),

        "Gradient Boosting (Tuned)":
            GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=RANDOM_STATE
            )
    }

    results = []

    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        results.append({

            "Model": name,

            "MAE":
                mean_absolute_error(
                    y_test,
                    predictions
                ),

            "MSE":
                mean_squared_error(
                    y_test,
                    predictions
                ),

            "RMSE":
                rmse,

            "R2":
                r2_score(
                    y_test,
                    predictions
                )
        })

        trained_models[name] = model

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\nRegression Results:")

    print(
        results_df.to_string(
            index=False
        )
    )

    best_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = (
        trained_models[best_name]
    )

    print(
        f"\nBest Regression Model: "
        f"{best_name}"
    )

    return (
        trained_models,
        results_df,
        best_name,
        best_model
    )


# ============================================================
# 4. REGRESSION HYPERPARAMETER TUNING
# ============================================================

def tune_regression_model(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n")
    print("=" * 60)
    print("REGRESSION HYPERPARAMETER TUNING")
    print("=" * 60)

    param_grid = {

        "n_estimators":
            [50, 100, 200],

        "learning_rate":
            [0.01, 0.1, 0.2],

        "max_depth":
            [3, 4, 5],

        "subsample":
            [0.8, 1.0]
    }

    model = GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=3,

        scoring="neg_mean_squared_error",

        n_jobs=-1,

        verbose=1
    )

    grid.fit(
        X_train,
        y_train
    )

    best_model = (
        grid.best_estimator_
    )

    predictions = (
        best_model.predict(X_test)
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nBest Regression Parameters:")

    print(
        grid.best_params_
    )

    print(
        f"Test MAE  : {mae:.4f}"
    )

    print(
        f"Test RMSE : {rmse:.4f}"
    )

    print(
        f"Test R2   : {r2:.4f}"
    )

    metrics = {

        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return (
        best_model,
        grid.best_params_,
        metrics
    )


# ============================================================
# 5. CLASSIFICATION MODELS
# ============================================================

def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),

        "Random Forest Classifier":
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting (Default)":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

        "Gradient Boosting (Tuned)":
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.01,
                max_depth=3,
                min_samples_split=10,
                subsample=0.9,
                random_state=RANDOM_STATE
            )
    }

    results = []

    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = (
            model.predict(X_test)
        )

        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        results.append({

            "Model": name,

            "Accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "ROC_AUC":
                roc_auc_score(
                    y_test,
                    probabilities
                )
        })

        trained_models[name] = model

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "ROC_AUC",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nClassification Results:")

    print(
        results_df.to_string(
            index=False
        )
    )

    best_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = (
        trained_models[best_name]
    )

    print(
        f"\nBest Classification Model: "
        f"{best_name}"
    )

    return (
        trained_models,
        results_df,
        best_name,
        best_model
    )


# ============================================================
# 6. CLASSIFICATION HYPERPARAMETER TUNING
# ============================================================

def tune_classification_model(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION HYPERPARAMETER TUNING")
    print("=" * 60)

    param_grid = {

        "n_estimators":
            [50, 100, 200],

        "learning_rate":
            [0.01, 0.1, 0.2],

        "max_depth":
            [3, 4, 5],

        "subsample":
            [0.8, 1.0]
    }

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=3,

        scoring="roc_auc",

        n_jobs=-1,

        verbose=1
    )

    grid.fit(
        X_train,
        y_train
    )

    best_model = (
        grid.best_estimator_
    )

    predictions = (
        best_model.predict(X_test)
    )

    probabilities = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    }

    print(
        "\nBest Classification Parameters:"
    )

    print(
        grid.best_params_
    )

    print("\nClassification Metrics:")

    for key, value in metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    return (
        best_model,
        grid.best_params_,
        metrics
    )


# ============================================================
# 7. SAVE SINGLE MODEL.PKL
# ============================================================

def save_models(
    regression_model,
    classification_model,
    features,
    regression_params,
    classification_params
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Store everything in one package
    # --------------------------------------------------------

    model_package = {

        "regression_model":
            regression_model,

        "classification_model":
            classification_model,

        "features":
            features,

        "regression_best_parameters":
            regression_params,

        "classification_best_parameters":
            classification_params
    }

    # --------------------------------------------------------
    # Save model.pkl
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "model.pkl"
    )

    joblib.dump(
        model_package,
        model_path
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "regression_target":
            "Next_Close",

        "classification_target":
            "Target",

        "classification_mapping": {

            "0":
                "DOWN / SELL",

            "1":
                "UP / BUY"
        },

        "regression_best_parameters":
            regression_params,

        "classification_best_parameters":
            classification_params
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n")
    print("=" * 60)
    print("MODELS SAVED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Model: {model_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("STOCK MARKET PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load and create features
    # --------------------------------------------------------

    df = create_features(
        DATA_PATH
    )

    print(
        f"\nFinal dataset shape: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
        features
    ) = prepare_data(df)

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows : "
        f"{len(X_test)}"
    )

    print(
        f"Number of features: "
        f"{len(features)}"
    )

    # ========================================================
    # REGRESSION
    # ========================================================

    (
        reg_models,
        reg_results,
        best_reg_name,
        default_best_reg
    ) = train_regression_models(

        X_train,
        X_test,

        y_reg_train,
        y_reg_test
    )

    # --------------------------------------------------------
    # Regression tuning
    # --------------------------------------------------------

    (
        tuned_reg_model,
        reg_params,
        reg_metrics
    ) = tune_regression_model(

        X_train,
        y_reg_train,

        X_test,
        y_reg_test
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    (
        clf_models,
        clf_results,
        best_clf_name,
        default_best_clf
    ) = train_classification_models(

        X_train,
        X_test,

        y_clf_train,
        y_clf_test
    )

    # --------------------------------------------------------
    # Classification tuning
    # --------------------------------------------------------

    (
        tuned_clf_model,
        clf_params,
        clf_metrics
    ) = tune_classification_model(

        X_train,
        y_clf_train,

        X_test,
        y_clf_test
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_models(

        regression_model=
            tuned_reg_model,

        classification_model=
            tuned_clf_model,

        features=
            features,

        regression_params=
            reg_params,

        classification_params=
            clf_params
    )

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    last_row = X_test.iloc[[-1]]

    # --------------------------------------------------------
    # Next Close
    # --------------------------------------------------------

    next_close_prediction = (
        tuned_reg_model
        .predict(last_row)[0]
    )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    class_prediction = int(

        tuned_clf_model
        .predict(last_row)[0]
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = (
        tuned_clf_model
        .predict_proba(last_row)[0]
    )

    down_probability = float(
        probabilities[0]
    )

    up_probability = float(
        probabilities[1]
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal = (
        "BUY"
        if class_prediction == 1
        else "SELL"
    )

    # ========================================================
    # DISPLAY FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("NEXT DAY PREDICTION")
    print("=" * 60)

    print(
        f"Predicted Next Close : "
        f"{next_close_prediction:.2f}"
    )

    print(
        f"Predicted Target     : "
        f"{class_prediction}"
    )

    print(
        f"Down Probability     : "
        f"{down_probability:.4f}"
    )

    print(
        f"Up Probability       : "
        f"{up_probability:.4f}"
    )

    print(
        f"Signal               : "
        f"{signal}"
    )

    print("\n0 = DOWN / SELL")
    print("1 = UP / BUY")

    print("\nTraining completed successfully.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
    
