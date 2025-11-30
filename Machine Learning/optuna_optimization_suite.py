"""
Suite completa de optimización con Optuna para modelos inmobiliarios
Incluye optimización multi-objetivo y análisis de importancia de parámetros
"""

import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiObjectiveOptimizer:
    """
    Optimizador multi-objetivo que balancea RMSE, velocidad de inferencia y estabilidad
    """
    
    def __init__(self, model_type='lightgbm', n_folds=5):
        self.model_type = model_type
        self.n_folds = n_folds
        self.cv = TimeSeriesSplit(n_splits=n_folds)
        self.best_study = None
        self.best_params = None
        
    def objective_lightgbm_multi(self, trial, X, y):
        """
        Función objetivo multi-criterio para LightGBM
        Optimiza: RMSE (primario), velocidad (secundario), estabilidad (terciario)
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 15),  # -1 = no limit
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'verbosity': -1,
            'n_jobs': -1
        }
        
        # Cross-validation con métricas múltiples
        rmse_scores = []
        training_times = []
        prediction_times = []
        
        for train_idx, val_idx in self.cv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            # Medir tiempo de entrenamiento
            start_time = datetime.now()
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50)],
                verbose_eval=False
            )
            training_time = (datetime.now() - start_time).total_seconds()
            training_times.append(training_time)
            
            # Medir tiempo de predicción
            start_time = datetime.now()
            predictions = model.predict(X_val_fold)
            prediction_time = (datetime.now() - start_time).total_seconds()
            prediction_times.append(prediction_time)
            
            # Calcular RMSE
            rmse = np.sqrt(mean_squared_error(y_val_fold, predictions))
            rmse_scores.append(rmse)
        
        # Métricas agregadas
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)  # Estabilidad
        mean_train_time = np.mean(training_times)
        mean_pred_time = np.mean(prediction_times)
        
        # Penalización por inestabilidad y lentitud
        stability_penalty = std_rmse / mean_rmse  # CV stability
        speed_penalty = np.log1p(mean_train_time + mean_pred_time * 100)  # Pred time más importante
        
        # Objetivo combinado (pesos ajustables)
        combined_objective = mean_rmse * (1 + 0.1 * stability_penalty + 0.05 * speed_penalty)
        
        # Reportar métricas separadas para análisis
        trial.set_user_attr('mean_rmse', mean_rmse)
        trial.set_user_attr('std_rmse', std_rmse)
        trial.set_user_attr('mean_train_time', mean_train_time)
        trial.set_user_attr('mean_pred_time', mean_pred_time)
        trial.set_user_attr('stability_penalty', stability_penalty)
        
        return combined_objective
    
    def objective_xgboost(self, trial, X, y):
        """
        Optimización específica para XGBoost con parámetros avanzados
        """
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=self.cv, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()
    
    def objective_ensemble(self, trial, X, y):
        """
        Optimización de ensemble con pesos dinámicos
        """
        # Parámetros para LightGBM
        lgb_params = {
            'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 200),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        }
        
        # Parámetros para XGBoost
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        }
        
        # Pesos del ensemble
        lgb_weight = trial.suggest_float('lgb_weight', 0.0, 1.0)
        xgb_weight = 1.0 - lgb_weight
        
        rmse_scores = []
        
        for train_idx, val_idx in self.cv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar LightGBM
            lgb_model = lgb.LGBMRegressor(**lgb_params, verbosity=-1, random_state=42)
            lgb_model.fit(X_train_fold, y_train_fold)
            lgb_pred = lgb_model.predict(X_val_fold)
            
            # Entrenar XGBoost  
            xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
            xgb_model.fit(X_train_fold, y_train_fold)
            xgb_pred = xgb_model.predict(X_val_fold)
            
            # Ensemble prediction
            ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
            
            rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
    
    def optimize_advanced(self, X, y, n_trials=200, timeout=3600):
        """
        Optimización avanzada con diferentes estrategias
        """
        results = {}
        
        # 1. Optimización LightGBM multi-objetivo
        print("Optimizando LightGBM (multi-objetivo)...")
        study_lgb = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        )
        
        study_lgb.optimize(
            lambda trial: self.objective_lightgbm_multi(trial, X, y),
            n_trials=n_trials//3,
            timeout=timeout//3,
            show_progress_bar=True
        )
        
        results['lightgbm'] = {
            'best_params': study_lgb.best_params,
            'best_value': study_lgb.best_value,
            'study': study_lgb
        }
        
        # 2. Optimización XGBoost
        print("Optimizando XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(
            lambda trial: self.objective_xgboost(trial, X, y),
            n_trials=n_trials//3,
            timeout=timeout//3,
            show_progress_bar=True
        )
        
        results['xgboost'] = {
            'best_params': study_xgb.best_params,
            'best_value': study_xgb.best_value,
            'study': study_xgb
        }
        
        # 3. Optimización Ensemble
        print("Optimizando Ensemble...")
        study_ensemble = optuna.create_study(direction='minimize')
        study_ensemble.optimize(
            lambda trial: self.objective_ensemble(trial, X, y),
            n_trials=n_trials//3,
            timeout=timeout//3,
            show_progress_bar=True
        )
        
        results['ensemble'] = {
            'best_params': study_ensemble.best_params,
            'best_value': study_ensemble.best_value,
            'study': study_ensemble
        }
        
        # Encontrar mejor modelo general
        best_model_type = min(results.keys(), key=lambda k: results[k]['best_value'])
        self.best_study = results[best_model_type]['study']
        self.best_params = results[best_model_type]['best_params']
        
        print(f"\nMejor modelo: {best_model_type}")
        print(f"Mejor RMSE: {results[best_model_type]['best_value']:.4f}")
        print(f"Mejores parámetros: {self.best_params}")
        
        return results
    
    def analyze_optimization(self, results):
        """
        Análisis detallado de la optimización
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (model_type, result) in enumerate(results.items()):
            study = result['study']
            
            # Plot optimization history
            ax = axes[i//2, i%2]
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax)
            ax.set_title(f'{model_type.upper()} - Optimization History')
        
        plt.tight_layout()
        plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Parameter importance analysis
        for model_type, result in results.items():
            study = result['study']
            try:
                importance = optuna.importance.get_param_importances(study)
                print(f"\n{model_type.upper()} - Parameter Importance:")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {param}: {imp:.4f}")
            except Exception as e:
                print(f"Could not analyze importance for {model_type}: {e}")

class ProductionModelTrainer:
    """
    Entrenador de modelos para producción con validación robusta
    """
    
    def __init__(self):
        self.optimizer = MultiObjectiveOptimizer()
        self.trained_models = {}
        
    def train_production_models(self, X, y, save_path="models/"):
        """
        Entrena y valida modelos optimizados para producción
        """
        print("Iniciando optimización de modelos para producción...")
        
        # Optimización completa
        results = self.optimizer.optimize_advanced(X, y, n_trials=150, timeout=7200)  # 2 horas max
        
        # Entrenar modelos finales con mejores parámetros
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        for model_type, result in results.items():
            print(f"\nEntrenando modelo final: {model_type}")
            
            if model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**result['best_params'], random_state=42)
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(**result['best_params'], random_state=42)
            elif model_type == 'ensemble':
                # Implementar ensemble específico
                model = self._create_ensemble_model(result['best_params'])
            
            model.fit(X, y)
            
            # Validación cruzada final
            cv_scores = cross_val_score(model, X, y, cv=self.optimizer.cv, 
                                      scoring='neg_root_mean_squared_error')
            
            model_info = {
                'model': model,
                'best_params': result['best_params'],
                'cv_rmse_mean': -cv_scores.mean(),
                'cv_rmse_std': cv_scores.std(),
                'optimization_study': result['study']
            }
            
            self.trained_models[model_type] = model_info
            
            # Guardar modelo
            model_path = f"{save_path}optimized_{model_type}_{timestamp}.joblib"
            joblib.dump(model_info, model_path)
            print(f"Modelo guardado: {model_path}")
            print(f"CV RMSE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Análisis de optimización
        self.optimizer.analyze_optimization(results)
        
        return self.trained_models
    
    def _create_ensemble_model(self, params):
        """
        Crea modelo ensemble con parámetros optimizados
        """
        # Separar parámetros por modelo
        lgb_params = {k.replace('lgb_', ''): v for k, v in params.items() if k.startswith('lgb_')}
        xgb_params = {k.replace('xgb_', ''): v for k, v in params.items() if k.startswith('xgb_')}
        weights = {'lgb': params['lgb_weight'], 'xgb': 1 - params['lgb_weight']}
        
        # Implementar clase ensemble personalizada aquí
        class EnsembleModel:
            def __init__(self, lgb_params, xgb_params, weights):
                self.lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=42)
                self.xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
                self.weights = weights
                
            def fit(self, X, y):
                self.lgb_model.fit(X, y)
                self.xgb_model.fit(X, y)
                return self
                
            def predict(self, X):
                lgb_pred = self.lgb_model.predict(X)
                xgb_pred = self.xgb_model.predict(X)
                return self.weights['lgb'] * lgb_pred + self.weights['xgb'] * xgb_pred
        
        return EnsembleModel(lgb_params, xgb_params, weights)

# Ejemplo de uso completo
if __name__ == "__main__":
    # Cargar datos (asumiendo que están preprocessados)
    df = pd.read_csv("tmp/ml_ready_13022025.csv")
    
    # Preparar features y target
    X = df.drop(['price'], axis=1)
    y = df['price']
    
    # Entrenar modelos optimizados
    trainer = ProductionModelTrainer()
    models = trainer.train_production_models(X, y)
    
    print("\n" + "="*50)
    print("RESUMEN DE OPTIMIZACIÓN COMPLETADA")
    print("="*50)
    
    for model_type, info in models.items():
        print(f"{model_type.upper()}:")
        print(f"  RMSE: {info['cv_rmse_mean']:,.2f} (±{info['cv_rmse_std']:.2f})")
        print(f"  Mejores parámetros: {len(info['best_params'])} parámetros optimizados")
        print()