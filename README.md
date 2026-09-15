# 📈 AI-Powered Stock Market Prediction & Portfolio Risk Management

## 📌 Project Overview

This project develops an **AI-based stock market analysis and prediction system** using historical stock prices, technical indicators, trading volume, volatility, and **news sentiment analysis**.

The system performs data cleaning, SQL-based analysis, exploratory data analysis, feature engineering, machine learning-based stock price prediction, and portfolio analysis.

The main objective is to **predict the next day's closing price** and analyze stock performance and investment risk.

---

## 🎯 Objectives

* Clean and preprocess stock market and news data.
* Analyze stock price movements and trading patterns.
* Extract useful information from news sentiment.
* Create technical and time-series features.
* Predict the **next-day closing price**.
* Compare different machine learning regression models.
* Analyze stock returns, volatility, trading volume, and sentiment.
* Support portfolio performance and risk analysis.

---

## 📊 Dataset

The project uses a stock market and news sentiment dataset containing **6,000 rows and 11 columns**.

### Main Features

| Feature           | Description           |
| ----------------- | --------------------- |
| `Date`            | Trading date          |
| `Symbol`          | Stock symbol          |
| `Open`            | Opening price         |
| `High`            | Highest price         |
| `Low`             | Lowest price          |
| `Close`           | Closing price         |
| `Volume`          | Trading volume        |
| `News_Headline`   | Related news headline |
| `Sentiment_Score` | News sentiment score  |
| `Volatility`      | Stock volatility      |
| `Target`          | Stock movement target |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Converted `Date` into the correct datetime format.
* Handled missing values.
* Removed duplicate records.
* Checked invalid values.
* Standardized numerical data.
* Sorted stock records chronologically by `Symbol` and `Date`.

---

## 🔧 Feature Engineering

Several time-series and technical features were created:

* `Return_1D`
* `Return_3D`
* `Return_5D`
* `SMA_5`
* `SMA_20`
* `Price_to_SMA20`
* `Volatility_5D`
* `Close_Lag_1`
* `Close_Lag_2`
* `Close_Lag_3`
* `Sentiment_Lag_1`
* `Sentiment_Lag_2`
* `Sentiment_Lag_3`
* `Sentiment_SMA_3D`

The target variable used for regression is:

**`Next_Close`**** — the next day's closing price.**

After feature generation and removing rows with missing values, the dataset contained **5,230 rows and 50 columns**.

---

## 🤖 Machine Learning Models

Three regression models were trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

An **80% training and 20% testing time-series split** was used to preserve chronological order.

---

## 📏 Model Evaluation

The models were evaluated using:

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **R² Score** — Coefficient of Determination

### Results

| Model                       |   RMSE |   MAE |   R² Score |
| --------------------------- | -----: | ----: | ---------: |
| Linear Regression           | 108.61 | 47.76 | **0.9969** |
| Random Forest Regressor     | 137.75 | 60.74 |     0.9950 |
| Gradient Boosting Regressor | 132.88 | 59.31 |     0.9953 |

Based on the recorded results, **Linear Regression achieved the best performance** among the three tested models.

---

## 📈 Prediction Visualization

The project compares:

* Actual next-day closing prices
* Linear Regression predictions
* Random Forest predictions
* Gradient Boosting predictions

It also generates a **feature importance analysis** using the Gradient Boosting model.

---

## 💼 Portfolio Analysis

The project also performs portfolio-level stock analysis using metrics such as:

* Average Daily Return
* Standard Deviation of Daily Returns
* Average Closing Price
* Average Sentiment Score
* Total Trading Volume
* Number of Trading Days
* Up-Movement Ratio

This helps compare stocks based on their **performance, market activity, sentiment, and movement patterns**.

---

## 🗂️ Project Workflow

```text
Raw Stock & News Data
        ↓
Data Cleaning
        ↓
SQL Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Time-Series Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Next-Day Closing Price Prediction
        ↓
Portfolio Performance & Risk Analysis
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **SQLite**
* **Google Colab / Jupyter Notebook**

---

## 📁 Project Structure

```text
stock-market-prediction/
│
├── data/
│   └── stock_news_11_columns_6000rows.csv
│
├── notebooks/
│   └── stock_prediction_project.ipynb
│
├── models/
│   └── trained_models.pkl
│
├── visualizations/
│   ├── regression_predictions_comparison.png
│   └── regression_feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-market-prediction.git
```

### 2. Open the Project

```bash
cd stock-market-prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the Notebook

Open:

```text
stock_prediction_project.ipynb
```

using **Jupyter Notebook** or **Google Colab**.

---

## 📌 Key Results

* The dataset was transformed into time-series features for prediction.
* **5,230 rows and 50 columns** remained after feature generation and NaN removal.
* Three regression models were compared.
* Linear Regression achieved the highest recorded **R² score of 0.9969**.
* Portfolio analysis was performed using return, volatility, sentiment, volume, and price-related metrics.

---

## ⚠️ Disclaimer

This project is created for **educational and research purposes only**.

Stock market predictions are uncertain and should not be considered financial advice. Past market data and model predictions do not guarantee future returns.

---

## 👨‍💻 Author

**BINSHAD**

Machine Learning | Data Science | Python | Stock Market Analytics

---

## ⭐ If You Find This Project Useful

If this project helps you, consider giving the repository a ⭐ on GitHub.
