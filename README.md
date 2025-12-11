# 🚗 **Ultimate Car Price Prediction Framework (UCPPF)**

**Version:** 2.0.2 (Enterprise Edition)

This project is a comprehensive, end-to-end machine learning pipeline designed to predict used car prices. It goes beyond simple model training by integrating data validation, advanced automated feature engineering, robust logging, error handling, and a Streamlit-based interactive web interface into a single, modular framework.

## 🚀 **Key Features**

* **Enterprise Architecture:** Built using software design patterns like **Singleton** (Configuration), **Factory** (Model Selection), and **Strategy** (Imputation).
* **Dual Interface:**
    * 🖥️ **CLI Mode:** Terminal-based menu for training, testing, and quick inference.
    * 🌐 **Web UI:** Modern dashboard built with **Streamlit** for visual analysis and user-friendly predictions.
* **Automated Feature Engineering:** Automatically generates new features (e.g., *Car Age*, *Usage Intensity*, *Segment*) and handles outliers.
* **Robust Logging:** Detailed, UTF-8 compatible logging system that records operations to both console and files.
* **Resilience:** Includes a **Mock Data Generator** that acts as a fallback if the dataset is missing, ensuring the code is always testable.
* **Model Versioning:** Saves trained models with associated metadata (metrics, timestamp) for reproducibility.

## 📂 **Dataset Information**

The repository includes the **`car_price_dataset.csv`** file required to train the model. You do not need to download external data; the pipeline is ready to run out of the box.

> **Note:** While the system is capable of generating synthetic (mock) data for testing purposes if the file is missing, it is recommended to keep the provided CSV file in the root directory for accurate real-world training.

## 🛠️ **Installation**

Ensure you have **Python 3.8+** installed.

### 1. Clone the repository:
```bash
git clone [https://github.com/ZeroPyon/CarPricePro.git](https://github.com/ZeroPyon/CarPricePro.git)

```

### 2\. Install dependencies:

You need Python installed. This project uses `streamlit`, `xgboost` and `scikit-learn`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost streamlit
```

## ▶️ **Usage**

You can run the application in two modes depending on your needs:

### **1. Web Interface (Recommended)**

Launch the interactive dashboard to predict prices via a GUI. This will open the app in your default browser.

```bash
streamlit run main.py
```

### **2. Command Line Interface (CLI)**

Run the script directly to access the maintenance menu and training options:

```bash
python main.py
```

## 🎛️ **CLI Menu Options**

When running in CLI mode, you will be presented with the following options:

| Key | Action | Description |
| :--- | :--- | :--- |
| **[1]** | **Retrain Model** | Triggers the training pipeline. You can choose between Random Forest, Gradient Boosting, or XGBoost. |
| **[2]** | **Quick Prediction** | Allows you to input car details manually in the terminal for an instant price estimate. |
| **[3]** | **Run Tests** | Executes the `TestSuite` to verify data generation, feature engineering, and model integrity. |
| **[4]** | **Launch Web UI** | Switches from CLI to the Streamlit Web Interface. |
| **[Q]** | **Quit** | Exits the application. |

## 🏗️ **Project Architecture**

The codebase is organized into modular classes for maintainability and scalability:

  * **AppConfig:** Singleton class for global settings and path management.
  * **ModelFactory:** Implements the *Factory Pattern* to instantiate models dynamically (RF, GBM, XGBoost).
  * **FeatureEngineer:** Handles automated transformations, segment creation, and outlier removal.
  * **PredictionService:** Manages the inference logic for end-users.
  * **WebUI:** Encapsulates the Streamlit frontend logic and visualization.

## 💻 **Technologies Used**

  * **Python 3.8+**
  * **Streamlit** (Interactive Web UI)
  * **Scikit-Learn** (Machine Learning Pipeline)
  * **XGBoost** (Gradient Boosting Framework)
  * **Pandas & NumPy** (Data Manipulation)

-----

## 📜 **License**

This project is licensed under the **MIT License**.


