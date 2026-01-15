import requests
import json
import random
import time

URL = 'http://localhost:8000/predict'

def generate_fake_data():
    return [
        round(random.uniform(4.3, 7.9), 1),
        round(random.uniform(2.0, 4.4), 1),
        round(random.uniform(1.0, 6.9), 1),
        round(random.uniform(0.1, 2.5), 1)
    ]

print('Запуск нагрузочного тестирования API.')

for i in range(100):
    data = {'data': generate_fake_data()}
    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            print(f'Запрос {i+1}: Успешно')
        else:
            print(f'Запрос {i+1}: Ошибка со статусом {response.status_code}')
    except Exception as e:
        print(f'Запрос {i+1}: Не удалось установить соединение - {e}')
    
    time.sleep(0.1)

print('Тестирование завершено.')