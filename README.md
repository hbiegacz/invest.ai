# Invest.ai | ML-Powered Bitcoin Forecasting 💰📈🔮
[![Contributors](https://img.shields.io/github/contributors/hbiegacz/invest.ai?color=red)](https://github.com/hbiegacz/invest.ai/graphs/contributors)
[![Commit Activity](https://img.shields.io/badge/Commits-📈%20View%20Graph-orange)](https://github.com/hbiegacz/invest.ai/graphs/commit-activity)
[![Repo Size](https://img.shields.io/github/repo-size/hbiegacz/invest.ai?color=yellow)](https://github.com/hbiegacz/invest.ai)
[![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-~2.5k-green?logo=git)](https://github.com/hbiegacz/invest.ai)
[![License](https://img.shields.io/github/license/hbiegacz/invest.ai?color=blue)](LICENSE)

***Use the power of Machine Learning to assist you in your Bitcoin investing journey!***

Invest.ai lets you compare Bitcoin price predictions from multiple ML models (TFT, LSTM, Random Forest, Linear Regression) side-by-side for smarter decisions.
It also tracks key market factors like US GDP, unemployment rates, trading volumes of other cryptos, and more through simple Docker-run dashboards. 

<!-- 
## 🎥 Quick demo
-->

## 💻 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![DjangoREST](https://img.shields.io/badge/DJANGO-REST-ff1709?style=for-the-badge&logo=django&logoColor=white&color=ff1709&labelColor=gray)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🧩 Architecture
The project utilizes a containerized architecture to ensure seamless deployment and scalability.

- **Backend (Django REST Framework):** Fetches market data from external APIs (Binance, FRED, Coinmetrics, Stooq), processes it using Parquet files, and serves as the API for predictions and analytics.
- **Machine Learning (Python):** Includes models like **TFT**, **LSTM**, **Random Forest**, and **Linear Regression**. Uses advanced feature engineering and **SHAP analysis** to refine and interpret model results.
- **Frontend (React + Vite + TypeScript):** A modern dashboard for data visualization and comparing different model predictions side-by-side.

<!-- 


## 🤖🧠 Machine learning
Models used:
TFT
LSTM
Random Forest
Linear Regression
-->


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
