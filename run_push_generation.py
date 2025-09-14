#!/usr/bin/env python3
"""
Скрипт для генерации пуш-уведомлений через Ollama API
"""

import pandas as pd
import os
from push_generator_ollama import PushGeneratorOllama
from hybrid_recommender import HybridRecommender

def main():
    """Основная функция для генерации пуш-уведомлений"""
    
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ПУШ-УВЕДОМЛЕНИЙ ЧЕРЕЗ OLLAMA")
    print("=" * 60)
    
    # Проверяем наличие файла рекомендаций
    if not os.path.exists("recommendations.csv"):
        print("Ошибка: Файл 'recommendations.csv' не найден!")
        print("Сначала запустите: python run_recommendations.py")
        return
    
    # Загружаем рекомендации
    print("1. Загрузка рекомендаций...")
    try:
        recommendations_df = pd.read_csv("recommendations.csv")
        print(f"   Загружено {len(recommendations_df)} рекомендаций")
    except Exception as e:
        print(f"   Ошибка при загрузке рекомендаций: {e}")
        return
    
    # Загружаем данные клиентов для контекста
    print("2. Загрузка данных клиентов...")
    try:
        recommender = HybridRecommender()
        features_df = recommender.load_data()
        print(f"   Загружено {len(features_df)} клиентов")
    except Exception as e:
        print(f"   Ошибка при загрузке данных клиентов: {e}")
        return
    
    # Создаем генератор уведомлений
    print("3. Инициализация генератора уведомлений...")
    generator = PushGeneratorOllama()
    
    # Проверяем подключение к Ollama
    print("4. Проверка подключения к Ollama...")
    if generator.check_ollama_connection():
        print("   ✓ Ollama подключен, будет использоваться AI-генерация")
    else:
        print("   ⚠ Ollama недоступен, будут использоваться шаблоны")
    
    # Генерируем уведомления
    print("5. Генерация пуш-уведомлений...")
    try:
        notifications_df = generator.generate_batch_notifications(recommendations_df, features_df)
        print("   Уведомления сгенерированы успешно")
    except Exception as e:
        print(f"   Ошибка при генерации уведомлений: {e}")
        return
    
    # Сохраняем результаты
    print("6. Сохранение результатов...")
    try:
        generator.save_notifications(notifications_df, "push_notifications.csv")
    except Exception as e:
        print(f"   Ошибка при сохранении: {e}")
        return
    
    # Выводим статистику
    print("7. Статистика уведомлений:")
    generator.print_statistics(notifications_df)
    
    # Показываем примеры
    print("\n8. Примеры пуш-уведомлений:")
    print("-" * 40)
    for i, (_, notif) in enumerate(notifications_df.head(3).iterrows()):
        print(f"\nКлиент {notif['client_code']}: {notif['product']}")
        print(f"Уведомление: {notif['push_notification']}")
    
    print("\n" + "=" * 60)
    print("ПУШ-УВЕДОМЛЕНИЯ УСПЕШНО СГЕНЕРИРОВАНЫ")
    print("=" * 60)
    print("Созданные файлы:")
    print("- push_notifications.csv - итоговые пуш-уведомления")
    print("\nФормат CSV:")
    print("client_code,product,push_notification")

def test_single_client(client_id: int = 1):
    """Тестирует генерацию уведомления для одного клиента"""
    
    print(f"ТЕСТ ГЕНЕРАЦИИ ДЛЯ КЛИЕНТА {client_id}")
    print("=" * 40)
    
    # Загружаем данные
    recommender = HybridRecommender()
    features_df = recommender.load_data()
    
    # Получаем рекомендацию для клиента
    client_data = features_df[features_df['client_code'] == client_id]
    if client_data.empty:
        print(f"Клиент {client_id} не найден!")
        return
    
    client_data = client_data.iloc[0]
    
    # Получаем рекомендацию
    recommendations = recommender.hybrid_recommendation(features_df)
    client_rec = recommendations[recommendations['client_code'] == client_id].iloc[0]
    
    print(f"Клиент: {client_data['name']}")
    print(f"Рекомендуемый продукт: {client_rec['product']}")
    print(f"Уверенность: {client_rec['confidence']:.3f}")
    
    # Генерируем уведомление
    generator = PushGeneratorOllama()
    context = generator.create_client_context(client_data)
    push_text = generator.generate_push_notification(client_data['name'], client_rec['product'], context)
    
    print(f"\nКонтекст: {context}")
    print(f"Пуш-уведомление: {push_text}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Тестовый режим
        client_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        test_single_client(client_id)
    else:
        # Полный режим
        main()

