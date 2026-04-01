<div align="center">
  
# 🚢 Titanic Survival Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*A comprehensive End-to-End Machine Learning Pipeline for the classic Kaggle Titanic challenge.*

</div>

---

## 📖 About The Project

This project provides a robust, reproducible machine learning pipeline to predict passenger survival on the Titanic. It goes beyond a simple script by incorporating software engineering best practices, rigorous experiment tracking with **MLflow**, and intelligent feature engineering.

### 🌟 Key Features

- **Advanced Feature Engineering**: Extracted passenger titles, calculated family sizes, and engineered new features like `FarePerPerson`.
- **Experiment Tracking**: Integrated with **MLflow** to track hyperparameters, metrics (accuracy), and artifacts seamlessly.
- **Hyperparameter Tuning**: Explores multiple combinations of `max_depth` and `n_estimators` for a Random Forest model.
- **Automated Visualizations**: Automatically generates and logs feature importance plots to MLflow.
- **Scalable Structure**: Organized cleanly into logical folders (`src/`, `notebooks/`, `models/`, `data/`) for scale.

---

## 📁 Project Structure

```text
├── data/                   # Raw and processed datasets (ignored by git)
├── models/                 # Saved model `.pkl` files (ignored by git)
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── src/                    # Source code for utility functions (if any)
├── train.py                # Main training pipeline and MLflow integration
├── feature_importance.png  # Generated chart of feature importance
├── requirements.txt        # Detailed project dependencies
└── .gitignore              # Files/folders to ignore in tracking
```

---

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Training the Model

To run the pipeline, engineer features, and train the Random Forest models across the defined search space, simply run:

```bash
python train.py
```

*This will output accuracy scores for different hyperparameter runs and save the best models to the `models/` directory.*

### 2. Viewing Experiments in MLflow

This project uses a local SQLite backend for MLflow tracking. To view your runs, metrics, and logged feature importance charts, start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Built With

* **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
* **[Scikit-Learn](https://scikit-learn.org/)** - Machine learning modeling
* **[MLflow](https://mlflow.org/)** - Experiment tracking and model management
* **[Matplotlib](https://matplotlib.org/)** - Data visualization

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check [issues page](https://github.com/your-username/titanic-survival-prediction/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
