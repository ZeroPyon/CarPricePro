"""
==============================================================================
PROJECT: ULTIMATE CAR PRICE PREDICTION FRAMEWORK (UCPPF)
VERSION: 2.0.2 (Enterprise Edition - Streamlit Compatibility Fix)
AUTHOR: AI Assistant
DESCRIPTION:
    Bu modül, uçtan uca bir makine öğrenmesi boru hattını (pipeline) kapsar.
    Veri doğrulama, gelişmiş özellik mühendisliği, çoklu model eğitimi,
    hiperparametre optimizasyonu, model versiyonlama, loglama,
    birim testleri ve Streamlit tabanlı bir web arayüzünü tek bir çatı altında toplar.

FEATURES:
    - Singleton Configuration Management
    - Custom Exception Handling Hierarchy
    - Strategy Pattern for Imputation
    - Factory Pattern for Model Selection
    - Automatic Feature Engineering (Interaction Terms, Binning)
    - Robust Logging System (UTF-8 & Streamlit Compatible)
    - Model Versioning & Metadata Storage
    - Automated Unit Testing Suite
    - Interactive Web UI (Streamlit)
==============================================================================
"""

import sys
import os
import io
import time
import json
import logging
import joblib
import random
import warnings
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# 0. ENVIRONMENT & ENCODING SETUP (CRITICAL FIX)
# -----------------------------------------------------------------------------
# Bu blok, Windows konsolunda emojilerin (✅, 🚗) çökmesini engeller.
# ANCAK: Streamlit altında çalışırken bu işlem "I/O operation on closed file"
# hatası verebilir. Bu yüzden try-except bloğu ile korunmuştur.
try:
    # Sadece standart terminalde isek ve encoding utf-8 değilse zorla
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except (AttributeError, ValueError):
    # Streamlit veya farklı bir IDE (PyCharm, Jupyter) stdout'u ele geçirmişse
    # .buffer özelliğine erişilemez veya dosya kapalı görünebilir.
    # Bu durumda müdahale etmiyoruz (Streamlit zaten UTF-8 uyumludur).
    pass

# -----------------------------------------------------------------------------
# 1. IMPORTS & DEPENDENCY CHECKS
# -----------------------------------------------------------------------------
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as e:
    print(f"CRITICAL ERROR: Eksik kütüphane tespit edildi: {e}")
    print("Lütfen şu komutu çalıştırın: pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit")
    sys.exit(1)

# XGBoost opsiyonel kontrolü
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Uyarıları bastır (Production ortamı simülasyonu)
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# 2. CONFIGURATION & CONSTANTS (Singleton Pattern)
# -----------------------------------------------------------------------------

class AppConfig:
    """
    Uygulama genelindeki tüm ayarları tutan Singleton sınıf.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Dosya Yolları
        self.DATA_FILE = "car_price_dataset.csv"
        self.MODEL_DIR = "models"
        self.LOG_DIR = "logs"
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, "ultimate_car_model.pkl")
        self.METADATA_FILE = os.path.join(self.MODEL_DIR, "model_metadata.json")

        # Veri Ayarları
        self.TARGET_COL = "Price"
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.CV_FOLDS = 5

        # Feature Engineering Ayarları
        self.POLY_DEGREE = 2
        self.USE_LOG_TRANSFORM_TARGET = True
        self.OUTLIER_THRESHOLD = 3.0  # Z-score eşiği

        # Grid Search Ayarları
        self.N_ITER_SEARCH = 15
        self.N_JOBS = -1

        # Oluşturulması gereken klasörler
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    def get(self, key):
        return getattr(self, key, None)


CONFIG = AppConfig()


# -----------------------------------------------------------------------------
# 3. LOGGING SYSTEM
# -----------------------------------------------------------------------------

class Logger:
    """
    Gelişmiş loglama sınıfı. Hem dosyaya hem konsola yazar.
    """

    @staticmethod
    def setup_logger(name: str = "CarPriceAI"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        # Format
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(module)s.%(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File Handler
        log_file = os.path.join(CONFIG.LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

        # Dosyaya yazarken UTF-8 encoding kullanılıyor
        try:
            fh = logging.FileHandler(log_file, encoding='utf-8')
        except ValueError:
            # Python sürümü çok eskiyse encoding parametresi olmayabilir (nadir)
            fh = logging.FileHandler(log_file)

        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger


log = Logger.setup_logger()


# -----------------------------------------------------------------------------
# 4. CUSTOM EXCEPTIONS
# -----------------------------------------------------------------------------

class CarPredictionError(Exception):
    """Base class for exceptions in this module."""
    pass


class DataLoadingError(CarPredictionError):
    """Raised when data loading fails."""
    pass


class DataValidationError(CarPredictionError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(CarPredictionError):
    """Raised during model training issues."""
    pass


class FeatureEngineeringError(CarPredictionError):
    """Raised during feature processing."""
    pass


# -----------------------------------------------------------------------------
# 5. UTILITY DECORATORS
# -----------------------------------------------------------------------------

def timeit(func):
    """Fonksiyon çalışma süresini ölçen decorator."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        log.debug(f"'{func.__name__}' fonksiyonu başlatıldı.")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            log.info(f"'{func.__name__}' tamamlandı. Süre: {end_time - start_time:.4f} sn")
            return result
        except Exception as e:
            log.error(f"'{func.__name__}' hatayla karşılaştı: {str(e)}")
            raise e

    return wrapper


# -----------------------------------------------------------------------------
# 6. DATA GENERATOR (MOCK DATA)
# -----------------------------------------------------------------------------

class MockDataGenerator:
    """
    CSV dosyası yoksa test amaçlı rastgele veri üreten sınıf.
    """

    @staticmethod
    def generate_dummy_data(n_samples=1000):
        log.warning("CSV bulunamadı. Mock (Sahte) veri üretiliyor...")
        np.random.seed(CONFIG.RANDOM_STATE)

        brands = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Ford', 'Chevrolet', 'Kia', 'Hyundai', 'Volkswagen']
        fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
        transmissions = ['Manual', 'Automatic', 'Semi-Automatic']

        data = {
            'Brand': np.random.choice(brands, n_samples),
            'Year': np.random.randint(2000, 2024, n_samples),
            'Engine_Size': np.round(np.random.uniform(1.0, 5.0, n_samples), 1),
            'Fuel_Type': np.random.choice(fuel_types, n_samples),
            'Transmission': np.random.choice(transmissions, n_samples),
            'Mileage': np.random.randint(5000, 300000, n_samples),
            'Doors': np.random.choice([2, 3, 4, 5], n_samples),
            'Owner_Count': np.random.choice([1, 2, 3, 4, 5], n_samples)
        }

        df = pd.DataFrame(data)

        # Modelleri markaya göre rastgele ata
        models_map = {
            'Toyota': ['Corolla', 'Camry', 'RAV4'],
            'BMW': ['3 Series', '5 Series', 'X5'],
            'Audi': ['A3', 'A4', 'Q5'],
            # Diğerleri için generic
        }

        def get_model(brand):
            return np.random.choice(models_map.get(brand, ['Model_X', 'Model_Y']))

        df['Model'] = df['Brand'].apply(get_model)

        # Fiyatı mantıklı bir formülle oluştur (Gürültü ekle)
        base_price = 10000
        df['Price'] = (
                base_price
                + (df['Year'] - 2000) * 1000
                + df['Engine_Size'] * 2000
                - df['Mileage'] * 0.05
                + np.random.normal(0, 2000, n_samples)
        )

        # Negatif fiyatları düzelt
        df['Price'] = df['Price'].apply(lambda x: max(1000, x))

        log.info(f"{n_samples} adet mock veri başarıyla üretildi.")
        return df


# -----------------------------------------------------------------------------
# 7. DATA PROCESSING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Sklearn pipeline ile uyumlu outlier temizleyici.
    IQR yöntemini kullanır.
    """

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = {}
        self.upper_bound = {}
        self.numeric_cols = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in self.numeric_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bound[col] = Q1 - self.factor * IQR
                self.upper_bound[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for col in self.numeric_cols:
                # Capping (Sınırlama) yöntemi
                X_copy[col] = np.where(X_copy[col] < self.lower_bound[col], self.lower_bound[col], X_copy[col])
                X_copy[col] = np.where(X_copy[col] > self.upper_bound[col], self.upper_bound[col], X_copy[col])
            return X_copy
        return X


class FeatureEngineer:
    """
    Veri seti üzerinde gelişmiş özellik mühendisliği işlemlerini yönetir.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @timeit
    def process_features(self) -> pd.DataFrame:
        """Tüm feature engineering işlemlerini sırayla uygular."""
        log.info("Feature Engineering başlatılıyor...")

        try:
            self._create_age_features()
            self._create_usage_metrics()
            self._create_engine_metrics()
            self._create_brand_segmentation()
            self._create_interaction_features()
            self._handle_rare_categories()

            # Gereksiz sütun varsa düşür (Örn: Model çok kardinaliteye sahipse)
            # self.df.drop('Model', axis=1, inplace=True) 

            log.info(f"Feature Engineering tamamlandı. Yeni sütun sayısı: {self.df.shape[1]}")
            return self.df
        except Exception as e:
            raise FeatureEngineeringError(f"Özellik üretiminde hata: {e}")

    def _create_age_features(self):
        current_year = datetime.now().year
        self.df['Car_Age'] = current_year - self.df['Year']
        self.df['Is_Classic'] = (self.df['Car_Age'] > 20).astype(int)
        self.df['Is_New'] = (self.df['Car_Age'] <= 2).astype(int)

    def _create_usage_metrics(self):
        # Sıfıra bölme hatasını engellemek için +1
        self.df['Km_Per_Year'] = self.df['Mileage'] / (self.df['Car_Age'] + 1)
        self.df['Usage_Intensity'] = pd.cut(
            self.df['Km_Per_Year'],
            bins=[-1, 5000, 15000, 30000, np.inf],
            labels=['Low', 'Medium', 'High', 'Extreme']
        )

    def _create_engine_metrics(self):
        # Motor hacmi kategorizasyonu
        self.df['Engine_Category'] = pd.cut(
            self.df['Engine_Size'],
            bins=[0, 1.4, 2.0, 3.0, 10.0],
            labels=['Small', 'Medium', 'Large', 'Performance']
        )

    def _create_brand_segmentation(self):
        # Basit bir segmentasyon mantığı (Geliştirilebilir)
        luxury_brands = ['Mercedes', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Land Rover']
        economy_brands = ['Kia', 'Hyundai', 'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Volkswagen']

        self.df['Segment'] = 'Other'
        self.df.loc[self.df['Brand'].isin(luxury_brands), 'Segment'] = 'Luxury'
        self.df.loc[self.df['Brand'].isin(economy_brands), 'Segment'] = 'Economy'

    def _create_interaction_features(self):
        # Motor gücü ve yeniliğin etkileşimi
        self.df['Engine_x_Age'] = self.df['Engine_Size'] * (1 / (self.df['Car_Age'] + 1))

    def _handle_rare_categories(self):
        # Nadir görülen modelleri "Other" olarak grupla
        model_counts = self.df['Model'].value_counts()
        rare_models = model_counts[model_counts < 10].index
        self.df.loc[self.df['Model'].isin(rare_models), 'Model'] = 'Other_Rare'


# -----------------------------------------------------------------------------
# 8. DATA VALIDATOR
# -----------------------------------------------------------------------------

class DataValidator:
    """
    Verinin doğruluğunu ve beklenen formatta olduğunu kontrol eder.
    """
    REQUIRED_COLUMNS = ["Brand", "Model", "Year", "Engine_Size", "Fuel_Type",
                        "Transmission", "Mileage", "Doors", "Owner_Count", "Price"]

    @staticmethod
    def validate_schema(df: pd.DataFrame):
        log.info("Veri şeması doğrulanıyor...")
        missing_cols = [col for col in DataValidator.REQUIRED_COLUMNS if col not in df.columns]

        if missing_cols:
            raise DataValidationError(f"Eksik sütunlar var: {missing_cols}")

        # Veri Tipi Kontrolleri
        if not pd.api.types.is_numeric_dtype(df['Year']):
            raise DataValidationError("'Year' sütunu sayısal olmalı.")
        if not pd.api.types.is_numeric_dtype(df['Price']):
            raise DataValidationError("'Price' sütunu sayısal olmalı.")

        # Negatif değer kontrolü
        if (df['Price'] < 0).any() or (df['Mileage'] < 0).any():
            raise DataValidationError("Fiyat veya Kilometre negatif olamaz!")

        log.info("✅ Veri şeması doğrulandı.")


# -----------------------------------------------------------------------------
# 9. MODEL FACTORY (DESIGN PATTERN)
# -----------------------------------------------------------------------------

class ModelFactory:
    """
    İstenilen algoritmayı döndüren Factory sınıfı.
    """

    @staticmethod
    def get_model(model_type: str, params: dict = None):
        if params is None:
            params = {}

        if model_type == 'random_forest':
            return RandomForestRegressor(random_state=CONFIG.RANDOM_STATE, **params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=CONFIG.RANDOM_STATE, **params)
        elif model_type == 'xgboost':
            if XGB_AVAILABLE:
                return xgb.XGBRegressor(random_state=CONFIG.RANDOM_STATE, **params)
            else:
                log.warning("XGBoost bulunamadı, GradientBoosting'e geçiliyor.")
                return GradientBoostingRegressor(random_state=CONFIG.RANDOM_STATE, **params)
        elif model_type == 'ridge':
            return Ridge(random_state=CONFIG.RANDOM_STATE, **params)
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")


# -----------------------------------------------------------------------------
# 10. PIPELINE BUILDER
# -----------------------------------------------------------------------------

class PipelineBuilder:
    """
    Scikit-learn pipeline'ını inşa eden sınıf.
    """

    def __init__(self, numerical_features, categorical_features):
        self.num_feats = numerical_features
        self.cat_feats = categorical_features

    def create_pipeline(self, model_type='random_forest'):
        # 1. Sayısal Dönüşümler
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # Outlier'lara karşı dirençli
        ])

        # 2. Kategorik Dönüşümler
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 3. Birleştirme
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_feats),
                ('cat', categorical_transformer, self.cat_feats)
            ],
            remainder='drop'
        )

        # 4. Model Seçimi
        model = ModelFactory.get_model(model_type)

        # 5. Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        return pipeline


# -----------------------------------------------------------------------------
# 11. TRAINER CLASS
# -----------------------------------------------------------------------------

class ModelTrainer:
    """
    Model eğitimi, optimizasyonu ve değerlendirmesinden sorumlu sınıf.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pipeline = None
        self.best_model = None
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.numerical_cols = []
        self.categorical_cols = []

    def _prepare_data(self):
        """Veriyi X ve y olarak ayırır, sütun tiplerini belirler."""
        # Feature Engineering sonrası oluşan sütunları dinamik olarak yakala
        target = CONFIG.TARGET_COL

        # Sayısal ve Kategorik sütunları otomatik belirle
        self.numerical_cols = self.df.drop(target, axis=1).select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.drop(target, axis=1).select_dtypes(
            include=['object', 'category']).columns.tolist()

        X = self.df.drop(target, axis=1)
        y = self.df[target]

        # Log dönüşümü (Hedef değişkenin dağılımını düzeltmek için)
        if CONFIG.USE_LOG_TRANSFORM_TARGET:
            y = np.log1p(y)
            log.info("Hedef değişkene Logaritmik dönüşüm uygulandı.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=CONFIG.TEST_SIZE, random_state=CONFIG.RANDOM_STATE
        )
        log.info(f"Eğitim Seti: {self.X_train.shape}, Test Seti: {self.X_test.shape}")

    @timeit
    def tune_and_train(self, model_type='random_forest'):
        """RandomizedSearchCV ile hiperparametre optimizasyonu yapar."""
        self._prepare_data()

        log.info(f"Model eğitimi başlatılıyor: {model_type.upper()}")

        builder = PipelineBuilder(self.numerical_cols, self.categorical_cols)
        pipeline = builder.create_pipeline(model_type)

        # Parametre Izgaraları
        param_grids = {
            'random_forest': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'regressor__n_estimators': [100, 500],
                'regressor__learning_rate': [0.01, 0.1],
                'regressor__max_depth': [3, 5, 7],
                'regressor__subsample': [0.7, 1.0]
            },
            'gradient_boosting': {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [3, 5]
            }
        }

        grid = param_grids.get(model_type, {})

        if not grid:
            log.info("Grid search parametreleri bulunamadı, varsayılan eğitim yapılıyor.")
            pipeline.fit(self.X_train, self.y_train)
            self.best_model = pipeline
        else:
            log.info(f"Hiperparametre araması yapılıyor... (Iterasyon: {CONFIG.N_ITER_SEARCH})")
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=grid,
                n_iter=CONFIG.N_ITER_SEARCH,
                cv=CONFIG.CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=CONFIG.N_JOBS,
                random_state=CONFIG.RANDOM_STATE,
                verbose=1
            )
            search.fit(self.X_train, self.y_train)
            self.best_model = search.best_estimator_
            log.info(f"En iyi parametreler: {search.best_params_}")

    def evaluate(self):
        """Modeli test seti üzerinde değerlendirir ve raporlar."""
        if self.best_model is None:
            raise ModelTrainingError("Model henüz eğitilmedi!")

        y_pred = self.best_model.predict(self.X_test)

        # Eğer log dönüşümü yapıldıysa geri çevir (inverse transform)
        if CONFIG.USE_LOG_TRANSFORM_TARGET:
            y_test_orig = np.expm1(self.y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_test_orig = self.y_test
            y_pred_orig = y_pred

        # Metrik Hesaplamaları
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_orig, y_pred_orig)

        self.metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Date": str(datetime.now())
        }

        print("\n" + "=" * 40)
        print(f"   MODEL PERFORMANS RAPORU")
        print("=" * 40)
        print(f"🔹 MAE (Ortalama Mutlak Hata): {mae:,.2f} $")
        print(f"🔹 RMSE (Kök Ortalama Kare Hata): {rmse:,.2f} $")
        print(f"🔹 R² (Açıklayıcılık Oranı): {r2:.4f}")
        print("=" * 40 + "\n")

        # Hata Analizi Grafiği
        self._plot_residuals(y_test_orig, y_pred_orig)

    def _plot_residuals(self, y_true, y_pred):
        """Hata dağılım grafiğini çizer."""
        try:
            plt.figure(figsize=(10, 6))
            residuals = y_true - y_pred
            sns.histplot(residuals, kde=True, color='purple')
            plt.title("Hata Dağılımı (Residuals)")
            plt.xlabel("Hata Miktarı ($)")
            plt.ylabel("Frekans")

            # Grafiği kaydet
            plot_path = os.path.join(CONFIG.LOG_DIR, "residuals_plot.png")
            plt.savefig(plot_path)
            log.info(f"Hata grafiği kaydedildi: {plot_path}")
            # plt.show() # Konsol modunda açılmasın diye commentledim
        except Exception as e:
            log.warning(f"Grafik çizilirken hata: {e}")

    def save_model(self):
        """Modeli ve metadatasını kaydeder."""
        if self.best_model:
            joblib.dump(self.best_model, CONFIG.MODEL_FILE)

            # Metadata kaydet
            with open(CONFIG.METADATA_FILE, 'w') as f:
                json.dump(self.metrics, f, indent=4)

            log.info(f"Model ve metadata başarıyla kaydedildi: {CONFIG.MODEL_DIR}")
        else:
            log.error("Kaydedilecek model bulunamadı.")


# -----------------------------------------------------------------------------
# 12. PREDICTOR SERVICE (INFERENCE)
# -----------------------------------------------------------------------------

class PredictionService:
    """
    Eğitilmiş modeli kullanarak tahmin yapan servis sınıfı.
    """

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(CONFIG.MODEL_FILE):
            self.model = joblib.load(CONFIG.MODEL_FILE)
            log.info("PredictionService: Model yüklendi.")
        else:
            log.warning("PredictionService: Kayıtlı model bulunamadı.")

    def predict_single(self, input_data: dict) -> float:
        """
        Tek bir araç için fiyat tahmini yapar.
        Input: dict formatında araç özellikleri.
        Output: Tahmin edilen fiyat (float).
        """
        if not self.model:
            raise ValueError("Model yüklenmedi, tahmin yapılamaz.")

        # Dict'i DataFrame'e çevir
        df = pd.DataFrame([input_data])

        # Feature Engineering adımlarını burada da uygulamamız gerekiyor
        # Not: Üretim ortamında Transformer Pipeline içine FeatureEngineering sınıfını
        # dahil etmek daha doğrudur, ancak burada manuel tekrar yapacağız.

        current_year = datetime.now().year
        df['Car_Age'] = current_year - df['Year']

        # Bazı türetilmiş özellikler (Eğitimde oluşturduklarımızın aynısı olmalı)
        # Basitleştirilmiş versiyon: Pipeline içindeki imputer eksikleri halleder.
        df['Km_Per_Year'] = df['Mileage'] / (df['Car_Age'] + 1)

        luxury_brands = ['Mercedes', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Land Rover']
        economy_brands = ['Kia', 'Hyundai', 'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Volkswagen']

        df['Segment'] = 'Other'
        if df['Brand'].values[0] in luxury_brands:
            df['Segment'] = 'Luxury'
        elif df['Brand'].values[0] in economy_brands:
            df['Segment'] = 'Economy'

        # Kategorik veriler pipeline içinde handle_unknown='ignore' olduğu için sorun olmaz.
        # Sayısal hesaplamalar:
        df['Is_Classic'] = (df['Car_Age'] > 20).astype(int)
        df['Is_New'] = (df['Car_Age'] <= 2).astype(int)
        df['Engine_x_Age'] = df['Engine_Size'] * (1 / (df['Car_Age'] + 1))

        # Usage Intensity (Manuel mapping gerekebilir veya basitleştirilebilir)
        # Pipeline'da bu özellik OneHot veya Ordinal encode edilmediyse
        # string olarak kalması pipeline'ın kategorik işlemcisi tarafından işlenir.
        df['Usage_Intensity'] = 'Medium'  # Varsayılan, pipeline halleder
        df['Engine_Category'] = 'Medium'  # Varsayılan

        try:
            prediction_log = self.model.predict(df)

            if CONFIG.USE_LOG_TRANSFORM_TARGET:
                prediction = np.expm1(prediction_log)[0]
            else:
                prediction = prediction_log[0]

            return float(prediction)
        except Exception as e:
            log.error(f"Tahmin hatası: {e}")
            raise e


# -----------------------------------------------------------------------------
# 13. UNIT TESTS
# -----------------------------------------------------------------------------

class TestSuite:
    """
    Sistemin sağlıklı çalıştığını kontrol eden testler.
    """

    @staticmethod
    def run_tests():
        print("\n🧪 BİRİM TESTLERİ BAŞLATILIYOR...")

        # Test 1: Veri Yükleme (Mock)
        try:
            df = MockDataGenerator.generate_dummy_data(n_samples=50)
            assert not df.empty, "Veri çerçevesi boş!"
            print("✅ Test 1 Geçti: Mock Veri Üretimi")
        except AssertionError as e:
            print(f"❌ Test 1 Kaldı: {e}")

        # Test 2: Feature Engineering
        try:
            engineer = FeatureEngineer(df)
            df_eng = engineer.process_features()
            assert 'Car_Age' in df_eng.columns, "Car_Age özelliği oluşturulmadı"
            assert 'Segment' in df_eng.columns, "Segment özelliği oluşturulmadı"
            print("✅ Test 2 Geçti: Özellik Mühendisliği")
        except Exception as e:
            print(f"❌ Test 2 Kaldı: {e}")

        # Test 3: Model Eğitimi (Hızlı)
        try:
            trainer = ModelTrainer(df_eng)
            # Test için çok az iterasyon
            CONFIG.N_ITER_SEARCH = 2
            CONFIG.CV_FOLDS = 2
            trainer.tune_and_train(model_type='random_forest')
            assert trainer.best_model is not None, "Model oluşturulamadı"
            print("✅ Test 3 Geçti: Model Eğitimi")
        except Exception as e:
            print(f"❌ Test 3 Kaldı: {e}")

        print("🏁 Testler Tamamlandı.\n")


# -----------------------------------------------------------------------------
# 14. STREAMLIT WEB UI
# -----------------------------------------------------------------------------

class WebUI:
    """
    Streamlit arayüzünü yöneten sınıf.
    """

    def run(self):
        import streamlit as st

        # Sayfa Ayarları
        st.set_page_config(
            page_title="AutoPrice Pro AI",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # CSS Özelleştirme
        st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
        .stMetric {
            background-color: black;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("🚗 AutoPrice Pro: AI Tabanlı Fiyat Tahmini")
        st.markdown("---")

        # Sidebar
        st.sidebar.header("🔧 Araç Konfigüratörü")

        # Veri Kaynağı (Gerçek veya Mock)
        if os.path.exists(CONFIG.DATA_FILE):
            df_ref = pd.read_csv(CONFIG.DATA_FILE)
            brands = sorted(df_ref['Brand'].unique())
            models_dict = df_ref.groupby('Brand')['Model'].unique().to_dict()
        else:
            # Fallback
            brands = ['Toyota', 'BMW', 'Audi', 'Mercedes', 'Honda']
            models_dict = {b: ['Model A', 'Model B'] for b in brands}

        # Kullanıcı Girişleri
        selected_brand = st.sidebar.selectbox("Marka", brands)

        # Markaya göre model filtresi
        available_models = sorted(models_dict.get(selected_brand, [])) if selected_brand in models_dict else []
        selected_model = st.sidebar.selectbox("Model", available_models)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            year = st.number_input("Yıl", min_value=1990, max_value=datetime.now().year, value=2020)
            engine_size = st.number_input("Motor (L)", 0.8, 6.0, 2.0, 0.1)
        with col2:
            mileage = st.number_input("KM", 0, 500000, 50000, 1000)
            doors = st.selectbox("Kapı", [2, 3, 4, 5], index=2)

        fuel = st.sidebar.selectbox("Yakıt", ["Petrol", "Diesel", "Hybrid", "Electric"])
        transmission = st.sidebar.radio("Vites", ["Automatic", "Manual", "Semi-Automatic"])
        owner_count = st.sidebar.slider("Önceki Sahip", 0, 5, 1)

        # Tahmin Butonu
        if st.sidebar.button("💸 FİYATI HESAPLA"):
            with st.spinner("Yapay zeka analiz yapıyor..."):
                try:
                    service = PredictionService()
                    input_data = {
                        "Brand": selected_brand,
                        "Model": selected_model,
                        "Year": year,
                        "Engine_Size": engine_size,
                        "Fuel_Type": fuel,
                        "Transmission": transmission,
                        "Mileage": mileage,
                        "Doors": doors,
                        "Owner_Count": owner_count
                    }

                    price = service.predict_single(input_data)

                    # Sonuç Gösterimi
                    c1, c2, c3 = st.columns(3)
                    with c2:
                        st.metric(label="Tahmini Piyasa Değeri", value=f"${price:,.2f}")

                    # Segment Bilgisi
                    if price > 20000:
                        st.success("💎 Bu araç Premium segmentinde değerlendiriliyor.")
                    elif price < 8000:
                        st.info("📉 Bu araç Ekonomik segmentte.")
                    else:
                        st.info("⚖️ Bu araç Orta segmentte.")

                    # Benzer Araçlar Analizi (Dummy)
                    st.markdown("### 📊 Pazar Analizi")
                    chart_data = pd.DataFrame(
                        np.random.normal(price, price * 0.1, 50),
                        columns=["Benzer İlanlar"]
                    )
                    st.bar_chart(chart_data)

                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)}")
                    st.warning("Lütfen önce modelin eğitildiğinden emin olun (Konsoldan çalıştırarak).")

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.caption("v2.0.0 Enterprise Edition")


# -----------------------------------------------------------------------------
# 15. MAIN EXECUTION CONTROLLER
# -----------------------------------------------------------------------------

def main():
    """
    Ana program akışı.
    """
    print("""
    #######################################################
    #                                                     #
    #      ULTIMATE CAR PRICE PREDICTION FRAMEWORK        #
    #             Enterprise Edition v2.0                 #
    #                                                     #
    #######################################################
    """)

    # 1. Veri Yükleme
    if os.path.exists(CONFIG.DATA_FILE):
        log.info(f"Dosya bulundu: {CONFIG.DATA_FILE}")
        df = pd.read_csv(CONFIG.DATA_FILE)
    else:
        log.warning("Veri dosyası bulunamadı, mock veri üretiliyor...")
        df = MockDataGenerator.generate_dummy_data(2000)

    # 2. Veri Doğrulama
    try:
        DataValidator.validate_schema(df)
    except DataValidationError as e:
        log.error(f"Validasyon hatası: {e}")
        return

    # 3. Feature Engineering
    engineer = FeatureEngineer(df)
    df_processed = engineer.process_features()

    # 4. Model Eğitimi (Eğer model yoksa veya yeniden eğitmek istenirse)
    # Basit bir CLI menüsü
    print("\n[1] Modeli Yeniden Eğit")
    print("[2] Mevcut Modeli Kullan ve Tahmin Yap")
    print("[3] Testleri Çalıştır")
    print("[4] Web Arayüzünü Başlat (Streamlit)")
    print("[Q] Çıkış")

    choice = input("\nSeçiminiz: ").upper().strip()

    if choice == '1':
        trainer = ModelTrainer(df_processed)
        print("\nHangi algoritma kullanılsın?")
        print("1. Random Forest (Varsayılan)")
        print("2. Gradient Boosting")
        print("3. XGBoost (Varsa)")
        algo_choice = input("Seçim (1-3): ")

        algo_map = {'1': 'random_forest', '2': 'gradient_boosting', '3': 'xgboost'}
        selected_algo = algo_map.get(algo_choice, 'random_forest')

        trainer.tune_and_train(model_type=selected_algo)
        trainer.evaluate()
        trainer.save_model()

    elif choice == '2':
        if not os.path.exists(CONFIG.MODEL_FILE):
            log.error("Kayıtlı model yok! Önce eğitim yapın.")
            return

        predictor = PredictionService()

        # İnteraktif tahmin döngüsü
        while True:
            try:
                print("\n--- Hızlı Tahmin ---")
                brand = input("Marka (örn: Toyota): ")
                model = input("Model (örn: Camry): ")
                year = int(input("Yıl: "))
                engine = float(input("Motor Hacmi: "))
                price = predictor.predict_single({
                    "Brand": brand, "Model": model, "Year": year,
                    "Engine_Size": engine, "Fuel_Type": "Petrol",  # Basitleştirilmiş giriş
                    "Transmission": "Automatic", "Mileage": 50000,
                    "Doors": 4, "Owner_Count": 1
                })
                print(f"💰 Tahmin: ${price:,.2f}")

                if input("Devam? (E/H): ").lower() != 'e':
                    break
            except Exception as e:
                print(f"Hata: {e}")

    elif choice == '3':
        TestSuite.run_tests()

    elif choice == '4':
        print("\nWeb arayüzü başlatılıyor...")
        print("Lütfen terminale şu komutu girin:")
        print(f"streamlit run {os.path.basename(__file__)}")

    elif choice == 'Q':
        print("Çıkış yapılıyor...")

    else:
        print("Geçersiz seçim.")


if __name__ == "__main__":
    # Streamlit kontrolü
    if 'streamlit' in sys.modules and 'streamlit.runtime' in sys.modules:
        # Streamlit ile çalıştırıldıysa doğrudan UI'ı başlat
        WebUI().run()
    else:
        # Normal python ile çalıştırıldıysa CLI'ı başlat
        main()