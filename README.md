# Invest.ai | ML-Powered Bitcoin Forecasting 💰📈🔮
[![Contributors](https://img.shields.io/github/contributors/hbiegacz/invest.ai?color=red)](https://github.com/hbiegacz/invest.ai/graphs/contributors)
[![Commit Activity](https://img.shields.io/badge/Commits-📈%20View%20Graph-orange)](https://github.com/hbiegacz/invest.ai/graphs/commit-activity)
[![Repo Size](https://img.shields.io/github/repo-size/hbiegacz/invest.ai?color=yellow)](https://github.com/hbiegacz/invest.ai)
[![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-~9.5k-green?logo=git)](https://github.com/hbiegacz/invest.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-73%25-purple)](backend/htmlcov/index.html)

***Use the power of Machine Learning to assist you in your Bitcoin investing journey!***

Invest.ai lets you compare Bitcoin price predictions from multiple ML models (TFT, LSTM, Random Forest, Linear Regression) side-by-side for smarter decisions.
It also tracks key market factors like US GDP, unemployment rates, trading volumes of other cryptos, and more through simple Docker-run dashboards. 

## 🎥 Quick demo
> You can find a more detailed & descriptive demonstration of the project on youtube [here](https://www.youtube.com/watch?v=gqYQ3Df0xIU).
![Quick Demo](docs/quick_demo.gif)

## 💻 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![DjangoREST](https://img.shields.io/badge/DJANGO-REST-ff1709?style=for-the-badge&logo=django&logoColor=white&color=ff1709&labelColor=gray)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Binance](https://img.shields.io/badge/Binance-FCD535?style=for-the-badge&logo=binance&logoColor=white)
## 📊 Data Sources
The models are trained and updated using data from multiple professional sources:
- **Crypto Market Data (Binance)**: Real-time and historical prices (Open, Close, High, Low), trading volumes, and number of trades for BTC, ETH, BNB, and XRP.
- **Economic Indicators (FRED)**: Key US economic metrics including **GDP** and **Unemployment Rate**.
- **Market Benchmarks (Stooq)**: S&P 500 index data used as a traditional market correlate.
- **On-chain & Reference Rates (Coinmetrics)**: Additional crypto-asset reference rates for enhanced model context.

## 🧩 Architecture
The project utilizes a containerized architecture to ensure seamless deployment and scalability.

- **Backend (Django REST Framework):** Fetches market data from external APIs (Binance, FRED, Coinmetrics, Stooq), processes it using Parquet files, and serves as the API for predictions and analytics.
- **Machine Learning (Python):** Includes models like **TFT**, **LSTM**, **Random Forest**, and **Linear Regression**. Uses advanced feature engineering and **SHAP analysis** to refine and interpret model results.
- **Frontend (React + Vite + TypeScript):** A modern dashboard for data visualization and comparing different model predictions side-by-side.

## 🤖 Machine Learning
### Models & Metrics
We evaluate model performance using **MAE**, **MSE**, **RMSE**, and **R²**. The project compares:
- **TFT (Temporal Fusion Transformer)** - current top-performing model.
- **LSTM (Long Short-Term Memory)**
- **Random Forest** & **Linear Regression** (Lasso/Ridge)
- **Naive Baseline** (predicts zero return)

### Development Process
The modeling process evolved through several iterations:
1. **Baseline & Linear Models**: Initial tests with Linear Regression and Random Forest showed improvement over the naive baseline but revealed that simple models often struggle with market noise.
2. **Data Refining & Key Features**: We used SHAP analysis to identify which data actually helps the models. We removed "noisy" data like raw prices (which are too volatile) and monthly indicators that change too slowly. Instead, we focused on **price changes (log-returns)** and **trend indicators (EWM)**, along with total market activity.
3. **Advanced Time-Series Models**: To better understand how past events affect the future, we used models designed for sequences:
   - **LSTM**: Looks at the last 96 days of data to spot medium-term trends.
   - **TFT (Best Performer)**: Uses an "attention" mechanism to focus on the most important historical moments. It also accounts for weekly and yearly cycles, which helped it achieve the best results in our tests.

## 🧪 Testing & Coverage
The backend is tested using `pytest` with `pytest-cov` for measurement.
- **Current Coverage:** **73%** (Backend)
- **Tests Passed:** 48 cases covering views, services, and models.

To run tests and see the coverage report locally:
```bash
  # Inside backend directory (or via docker exec)
  pytest --cov=marketdata --cov-report=term-missing
```

Detailed HTML reports are generated in `backend/htmlcov/`.

## 🚀 Installation & Execution
Prerequisites:
- Docker Engine
- Docker Compose
> Note: Be aware that the initial installation process might take a while because all the dependencies are being downloaded.
```bash
  git clone https://github.com/hbiegacz/invest.ai.git
  cd invest.ai
  docker compose up --build
```
Open the web app http://localhost:5173
