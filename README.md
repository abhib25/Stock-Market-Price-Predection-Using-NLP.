# Predicting Stock Market Price Movements Based on News Headlines Using NLP

## Overview
This project aims to predict stock market price movements by analyzing financial news headlines using Natural Language Processing (NLP).
The model leverages FinBERT for sentiment analysis and integrates sentiment scores with numerical stock indicators such as historical prices and moving averages.
Supervised machine learning models like Logistic Regression and Random Forest Classifier are used to predict market trends.

## Features
- **Sentiment Analysis**: Uses FinBERT to classify financial news headlines as positive, neutral, or negative.
- **Feature Engineering**: Includes TF-IDF vectorization, moving averages (SMA, EMA), and stock market indicators.
- **Machine Learning Models**: Implements Logistic Regression and Random Forest for classification.
- **Evaluation Metrics**: Uses accuracy, precision, recall, and F1-score to compare model performances.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, TensorFlow, PyTorch
- **NLP Libraries**: Hugging Face Transformers, NLTK, SpaCy
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Dataset
The project uses stock market data from major indices such as the Dow Jones Industrial Average (DJIA) and S&P 500, along with financial news headlines from reputable sources. The dataset includes:
- **News Headlines**: Top 25 daily news headlines per trading day.
- **Stock Market Data**: Open, Close, High, Low prices, and Trading Volume.
- **Sentiment Scores**: Derived using FinBERT.

## Installation
To set up the project environment, follow these steps:
```bash
# Clone the repository
git clone https://github.com/stock-market-prediction-nlp.git
cd stock-market-prediction-nlp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Data Preprocessing
```python
python preprocess.py
```
Cleans financial news headlines, merges stock data, and applies feature engineering.

### 2. Sentiment Analysis
```python
python sentiment_analysis.py
```
Extracts sentiment scores from news headlines using FinBERT.

### 3. Train and Evaluate Models
```python
python train_model.py
```
Trains and evaluates Logistic Regression and Random Forest models.

### 4. Predict Stock Market Movements
```python
python predict.py --input "Federal Reserve raises interest rates"
```
Predicts whether a given news headline will impact stock prices positively or negatively.

## Results and Performance
The model evaluation indicates that:
- **Random Forest outperforms Logistic Regression**, capturing complex relationships in sentiment and price trends.
- **Sentiment alone is not a strong predictor**, but when combined with numerical indicators, prediction accuracy improves.
- **Evaluation Metrics**:
  - Logistic Regression: Accuracy ~ 65%
  - Random Forest: Accuracy ~ 72%

## Future Enhancements
- **Integrate deep learning models (LSTMs, Transformers) for time-series forecasting.**
- **Use real-time financial news for live stock market predictions.**
- **Explore additional macroeconomic indicators for better prediction accuracy.**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
**Abhishek Basavaraju**
Email: ab2249@hw.ac.uk  
