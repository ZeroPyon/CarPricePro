🚗 Ultimate Car Price Prediction Framework (UCPPF)
Version: 2.0.2 (Enterprise Edition)

This project is a comprehensive, end-to-end machine learning pipeline designed to predict used car prices. It goes beyond simple model training by integrating data validation, advanced automated feature engineering, robust logging, error handling, and a Streamlit-based interactive web interface into a single, modular framework.

🚀Key Features
Enterprise Architecture: Built using software design patterns like Singleton (Configuration), Factory (Model Selection), and Strategy (Imputation).

Dual Interface:

🖥 CLI Mode: Terminal-based menu for training, testing, and quick inference.

🌐 Web UI: Modern dashboard built with Streamlit for visual analysis and user-friendly predictions.

Automated Feature Engineering: Automatically generates new features (e.g., Car Age, Usage Intensity, Segment) and handles outliers.

Robust Logging: Detailed, UTF-8 compatible logging system that records operations to both console and files.

Resilience: Includes a Mock Data Generator that acts as a fallback if the dataset is missing, ensuring the code is always testable.

Model Versioning: Saves trained models with associated metadata (metrics, timestamp) for reproducibility.

📂 Dataset Information
The repository includes the car_price_dataset.csv file required to train the model. You do not need to download external data; the pipeline is ready to run out of the box.

Note: While the system is capable of generating synthetic (mock) data for testing purposes if the file is missing, it is recommended to keep the provided CSV file in the root directory for accurate real-world training.

🛠️ Installation
Ensure you have Python 3.8+ installed.

Clone the repository:
---

Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost streamlit

▶️ Usage
You can run the application in two modes:

1. Web Interface (Recommended)
Launch the interactive dashboard to predict prices via a GUI:
streamlit run main.py
This will open http://localhost:8501 in your default browser.

2. Command Line Interface (CLI)
Run the script directly to access the maintenance menu:
python main.py
The CLI menu allows you to:

[1] Retrain the model (Choose between Random Forest, XGBoost, etc.)

[2] Make quick predictions via terminal

[3] Run the Unit Test Suite

[4] Launch the Web UI

🏗️ Project Architecture
The codebase is organized into modular classes for maintainability:

AppConfig: Singleton class for global settings and paths.

DataValidator: Ensures input data schema and integrity.

FeatureEngineer: Handles transformations and domain-specific feature creation.

ModelFactory: Implements the Factory pattern to instantiate models (RF, GBM, XGBoost).

ModelTrainer: Manages the training pipeline and Hyperparameter Tuning (RandomizedSearchCV).

PredictionService: Handles inference logic for the end-user.

WebUI: Encapsulates the Streamlit frontend logic.

📝 License
This project is licensed under the MIT License.
