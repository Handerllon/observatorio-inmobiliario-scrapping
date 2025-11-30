"""
Algoritmo mejorado de feature selection para predicción inmobiliaria
Incluye feature engineering avanzado y selección automática con Optuna
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def engineer_features(self, df):
        """
        Feature engineering avanzado para propiedades inmobiliarias
        """
        df = df.copy()
        
        # 1. FEATURES DERIVADAS DE RATIOS
        df['price_per_m2'] = df['price'] / df['total_area']  # Métrica clave inmobiliaria
        df['room_density'] = df['rooms'] / df['total_area']   # Aprovechamiento del espacio
        df['bedroom_ratio'] = df['bedrooms'] / df['rooms']    # Distribución de ambientes
        
        # 2. FEATURES DE CALIDAD/LUJO
        df['luxury_score'] = (df['garages'] * 2 + df['bathrooms']) / df['rooms']
        df['total_facilities'] = df['garages'] + df['bathrooms']
        
        # 3. FEATURES DE EFICIENCIA ESPACIAL
        df['area_per_room'] = df['total_area'] / df['rooms']
        df['bathroom_adequacy'] = df['bathrooms'] / df['bedrooms'].clip(lower=1)
        
        # 4. FEATURES CATEGÓRICAS ORDINALES
        # Clasificación por tamaño
        df['size_category'] = pd.cut(df['total_area'], 
                                   bins=[0, 40, 70, 100, 200], 
                                   labels=['small', 'medium', 'large', 'xl'])
        
        # Clasificación por antigüedad
        df['age_category'] = pd.cut(df['antiquity'],
                                  bins=[-1, 5, 15, 30, 100],
                                  labels=['new', 'modern', 'established', 'vintage'])
        
        # 5. FEATURES DE INTERACCIÓN
        df['area_age_interaction'] = df['total_area'] * np.log1p(df['antiquity'])
        df['premium_location_indicator'] = (df['price'] > df['price'].quantile(0.75)).astype(int)
        
        return df
    
    def select_features_recursive(self, X, y, model=None):
        """
        Selección recursiva de features con cross-validation
        """
        if model is None:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            
        # RFECV con TimeSeriesSplit para validación temporal
        cv = TimeSeriesSplit(n_splits=5)
        rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        rfecv.fit(X, y)
        
        selected_features = X.columns[rfecv.support_]
        feature_scores = rfecv.cv_results_
        
        return selected_features, feature_scores, rfecv
    
    def statistical_feature_selection(self, X, y, k='all'):
        """
        Selección estadística basada en F-score
        """
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Obtener scores y p-values
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        feature_stats = pd.DataFrame({
            'feature': X.columns,
            'f_score': scores,
            'p_value': pvalues
        }).sort_values('f_score', ascending=False)
        
        return feature_stats, selector

class OptunaHyperparameterOptimizer:
    def __init__(self, model_type='lightgbm'):
        self.model_type = model_type
        self.best_params = None
        self.study = None
        
    def objective_lightgbm(self, trial, X_train, y_train, X_val, y_val):
        """
        Función objetivo para optimización de LightGBM con Optuna
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbosity': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                LightGBMPruningCallback(trial, 'valid_0-rmse')
            ],
            verbose_eval=False
        )
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        
        return rmse
    
    def objective_gradient_boosting(self, trial, X_train, y_train, cv_folds=5):
        """
        Función objetivo para optimización de Gradient Boosting
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        cv = TimeSeriesSplit(n_splits=cv_folds)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()  # Negativo porque Optuna minimiza
    
    def optimize(self, X_train, y_train, X_val=None, y_val=None, n_trials=100):
        """
        Ejecuta la optimización de hiperparámetros
        """
        study = optuna.create_study(direction='minimize')
        
        if self.model_type == 'lightgbm':
            if X_val is None or y_val is None:
                raise ValueError("LightGBM optimization requires validation set")
            
            study.optimize(
                lambda trial: self.objective_lightgbm(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True
            )
        elif self.model_type == 'gradient_boosting':
            study.optimize(
                lambda trial: self.objective_gradient_boosting(trial, X_train, y_train),
                n_trials=n_trials,
                show_progress_bar=True
            )
        
        self.study = study
        self.best_params = study.best_params
        
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value

class ImprovedRealEstatePredictor:
    def __init__(self, model_type='lightgbm'):
        self.model_type = model_type
        self.feature_engineer = AdvancedFeatureEngineer()
        self.optimizer = OptunaHyperparameterOptimizer(model_type)
        self.model = None
        self.selected_features = None
        
    def prepare_data(self, df):
        """
        Prepara datos con feature engineering completo
        """
        # Feature engineering
        df_engineered = self.feature_engineer.engineer_features(df)
        
        # Separar features categóricas para one-hot encoding
        categorical_features = ['size_category', 'age_category']
        df_categorical = pd.get_dummies(df_engineered[categorical_features], 
                                      prefix=categorical_features)
        
        # Combinar features numéricas y categóricas
        numerical_features = df_engineered.select_dtypes(include=[np.number]).columns
        numerical_features = numerical_features.drop('price')  # Excluir target
        
        X = pd.concat([
            df_engineered[numerical_features],
            df_categorical
        ], axis=1)
        
        y = df_engineered['price']
        
        return X, y
    
    def train_with_optimization(self, df, test_size=0.2, n_trials=100):
        """
        Entrena modelo con optimización completa de features e hiperparámetros
        """
        X, y = self.prepare_data(df)
        
        # Split temporal para real estate (más reciente = test)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Feature selection
        print("Ejecutando feature selection...")
        selected_features, _, rfecv = self.feature_engineer.select_features_recursive(X_train, y_train)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        self.selected_features = selected_features
        print(f"Features seleccionadas: {len(selected_features)}")
        print(selected_features.tolist())
        
        # Split para validación de Optuna
        val_split = int(len(X_train_selected) * 0.8)
        X_train_opt, X_val_opt = X_train_selected.iloc[:val_split], X_train_selected.iloc[val_split:]
        y_train_opt, y_val_opt = y_train.iloc[:val_split], y_train.iloc[val_split:]
        
        # Hyperparameter optimization
        print("Optimizando hiperparámetros...")
        best_params, best_score = self.optimizer.optimize(
            X_train_opt, y_train_opt, X_val_opt, y_val_opt, n_trials
        )
        
        # Entrenar modelo final
        if self.model_type == 'lightgbm':
            train_data = lgb.Dataset(X_train_selected, label=y_train)
            self.model = lgb.train(
                {**best_params, 'objective': 'regression', 'metric': 'rmse', 'verbosity': -1},
                train_data,
                num_boost_round=1000
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**best_params)
            self.model.fit(X_train_selected, y_train)
        
        # Evaluación final
        train_pred = self.predict(X_train_selected)
        test_pred = self.predict(X_test_selected)
        
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
        
        print(f"\nResultados finales:")
        print(f"Train RMSE: {train_rmse:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")
        print(f"Mejora estimada: {(1 - test_rmse/best_score)*100:.1f}%")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'best_params': best_params,
            'selected_features': selected_features
        }
    
    def predict(self, X):
        """
        Predicción con modelo entrenado
        """
        if self.selected_features is not None:
            X = X[self.selected_features]
            
        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def save_model(self, path):
        """
        Guarda modelo y metadata
        """
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'model_type': self.model_type,
            'best_params': self.optimizer.best_params
        }
        joblib.dump(model_data, path)
        print(f"Modelo guardado en: {path}")

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("tmp/ml_ready_13022025.csv")
    
    # Crear predictor mejorado
    predictor = ImprovedRealEstatePredictor(model_type='lightgbm')
    
    # Entrenar con optimización completa
    results = predictor.train_with_optimization(df, n_trials=50)
    
    # Guardar modelo optimizado
    predictor.save_model("models/improved_model_optimized.joblib")