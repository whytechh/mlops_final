import pandas as pd
import matplotlib.pyplot as plt
import os

# Сопоставление предсказаний между двумя моделями
def analyze_results():
    log_path = 'data/ab_logs.csv'
    
    if not os.path.exists(log_path):
        print(f'ОШИБКА Файл {log_path} не найден')
        return

    df = pd.read_csv(log_path)
    
    variant_col = 'variant' if 'variant' in df.columns else 'model_variant'
    
    print('A/B тестирование')
    print('-' * 30)
    print(f'Всего обработано запросов: {len(df)}')
    print(f'Столбец для целевого анализа: {variant_col}')
    
    # Статистическая группировка данных
    summary = df.groupby(variant_col)['prediction'].value_counts(normalize=True).unstack().fillna(0)
    
    print('\nРаспределение предсказаний:')
    print(summary)
    
    plt.figure(figsize=(10, 6))
    summary.plot(kind='bar', stacked=False)
    plt.title('Результаты A/B теста')
    plt.xlabel('Вариант модели')
    plt.ylabel('Доля предсказаний')
    plt.legend(title='Iris Classes', labels=['Setosa (0)', 'Versicolor (1)', 'Virginica (2)'])
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    report_path = 'data/ab_test.png'
    plt.savefig(report_path)
    print('-' * 30)
    print(f'Готово: Отчет сохранен по пути {report_path}')

if __name__ == '__main__':
    analyze_results()