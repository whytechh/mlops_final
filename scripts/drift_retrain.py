import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pycaret.classification import setup, compare_models
import mlflow
from mlflow.tracking import MlflowClient
import os

# Глобальные настройки системы
MLFLOW_URI = 'http://mlflow:5000'
MODEL_NAME = 'model_mlops'
ALIAS_CHALLENGER = 'challenger'

mlflow.set_tracking_uri(MLFLOW_URI)

# Предобработка имен признаков
def clean_column_names(df):
    # Применяем трансформацию ко всем заголовкам через метод map
    df.columns = df.columns.to_series().str.replace(' (cm)', '', regex=False).str.replace(' ', '_').str.lower()
    return df

# Использование теста Колмогорова-Смирнова для определения дрифта
def check_for_drift(reference_path='/opt/airflow/data/reference_data.csv', 
                    current_path='/opt/airflow/data/current_data.csv', 
                    threshold=0.05):
    print(f'Запуск мониторинга отклонений: {reference_path} vs {current_path}')
    
    # Чтение данных с немедленной очисткой колонок
    ref = clean_column_names(pd.read_csv(reference_path))
    cur = clean_column_names(pd.read_csv(current_path))
    
    # Формируем список признаков, отсеивая целевой столбец
    features = [c for c in ref.columns if c != 'target']
    
    # Вычисляем p-value для каждой колонки и проверяем значимость
    drifts = []
    for col in features:
        # Извлекаем только p-значение из результата теста
        p_value = ks_2samp(ref[col], cur[col])[1]
        drifts.append(p_value < 0.05)
    
    # Расчет среднего уровня дрифта по всем фичам
    drift_share = sum(drifts) / len(drifts) if drifts else 0
    print(f'Доля признаков с выявленным дрифтом: {drift_share:.2f}')
    
    return bool(drift_share > threshold)

# Переобучение и регистрация в Mlflow
def train_and_register_model(data_path='/opt/airflow/data/current_data.csv'):
    print('Запуск пайплайна переобучения в PyCaret...')
    df = clean_column_names(pd.read_csv(data_path))
    
    # Оборачиваем процесс в контекст MLflow для автоматического логирования
    with mlflow.start_run(run_name='retraining_run'):
        # Конфигурируем среду PyCaret
        s = setup(
            data=df, 
            target='target', 
            log_experiment=True, 
            experiment_name='iris_ab_test', 
            html=False, 
            verbose=False
        )
        
        # Запускаем кросс-валидацию и поиск лучшего алгоритма
        best_model = compare_models()
        print(f'Результат подбора модели: {best_model}')
        
        # Фиксация модели в реестре MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path='model', 
            registered_model_name=MODEL_NAME
        )
        
        # Работа с метаданными через MlflowClient
        client = MlflowClient()
        new_version = model_info.registered_model_version
        
        # Обновление алиаса для новой версии
        client.set_registered_model_alias(MODEL_NAME, ALIAS_CHALLENGER, str(new_version))
        print(f"Версии {new_version} присвоен статус '{ALIAS_CHALLENGER}'")
        
        return {'version': new_version, 'alias': ALIAS_CHALLENGER}

if __name__ == '__main__':
    # Выполняем проверку и принимаем решение о переобучении
    if check_for_drift():
        train_and_register_model()
    else:
        print('Дрифт не обнаружен. Текущая модель остается актуальной.')