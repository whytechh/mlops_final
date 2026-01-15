from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import sys

sys.path.append('/opt/airflow/scripts')
import drift_retrain

# Функция для выполнения задачи мониторинга дрифта
def _check_drift_task(**context):
    # Вызов внешней функции для анализа отклонений данных
    drift_detected = drift_retrain.check_for_drift()
    
    # Передача результата через Task Instance для последующего использования
    ti = context.get('ti')
    ti.xcom_push(key='drift_status', value=drift_detected)
    
    return drift_detected

# Определение логики перехода между ветками
def _branching_logic(**context):

    ti = context.get('ti')
    # Извлечение сохраненного статуса дрифта
    drift_detected = ti.xcom_pull(task_ids='check_drift', key='drift_status')
    
    # Если зафиксировано расхождение, активируем ветку переобучения
    if drift_detected:
        print('Обнаружен дрифт данных. Переход к задаче retraining.')
        return 'retraining'
    
    # В противном случае завершаем пайплайн через пропуск
    print('Данные в норме. Выполнение ветки skip.')
    return 'skip'

# Описание конфигурации и структуры DAG
with DAG(
    dag_id='check_drift',
    start_date=days_ago(1),
    schedule_interval='@daily',
    catchup=False,
    tags=['drift', 'retraining']
) as dag:

    # Входная точка процесса
    start = EmptyOperator(task_id='start')

    # Шаг оценки качества входных данных
    check_drift = PythonOperator(
        task_id='check_drift',
        python_callable=_check_drift_task
    )

    # Узел принятия решения о дальнейшем пути
    branching = BranchPythonOperator(
        task_id='XOR',
        python_callable=_branching_logic
    )

    # Исполнение процесса переобучения модели
    retrain = PythonOperator(
        task_id='retraining',
        python_callable=drift_retrain.train_and_register_model
    )

    # Фиктивный узел для случая отсутствия изменений
    skip = EmptyOperator(task_id='skip')

    # Финальная точка сбора всех веток
    end = EmptyOperator(
        task_id='end', 
        trigger_rule='none_failed_min_one_success'
    )

    # Определение последовательности выполнения
    start >> check_drift >> branching
    branching >> retrain >> end
    branching >> skip >> end
