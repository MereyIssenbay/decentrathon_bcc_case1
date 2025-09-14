#!/usr/bin/env python3
"""
Главный скрипт для запуска гибридной рекомендательной системы
"""

import pandas as pd
import os
from hybrid_recommender import HybridRecommender

def main():
    """Основная функция для запуска рекомендаций"""
    
    print("=" * 60)
    print("ГИБРИДНАЯ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
    print("=" * 60)
    
    # Проверяем наличие папки с данными
    if not os.path.exists("data"):
        print("Ошибка: Папка 'data' не найдена!")
        print("Убедитесь, что файлы данных находятся в папке 'data/'")
        return
    
    # Создаем рекомендательную систему
    print("1. Инициализация гибридной рекомендательной системы...")
    recommender = HybridRecommender()
    
    # Загружаем данные
    print("2. Загрузка данных...")
    try:
        features_df = recommender.load_data()
        print(f"   Загружено {len(features_df)} клиентов")
    except Exception as e:
        print(f"   Ошибка при загрузке данных: {e}")
        return
    
    # Получаем рекомендации
    print("3. Генерация рекомендаций...")
    try:
        recommendations = recommender.hybrid_recommendation(features_df)
        print("   Рекомендации сгенерированы успешно")
    except Exception as e:
        print(f"   Ошибка при генерации рекомендаций: {e}")
        return
    
    # Сохраняем результаты
    print("4. Сохранение результатов...")
    try:
        recommender.save_recommendations(recommendations, "recommendations.csv")
    except Exception as e:
        print(f"   Ошибка при сохранении: {e}")
        return
    
    # Выводим статистику
    print("5. Статистика рекомендаций:")
    recommender.print_statistics(recommendations)
    
    # Показываем примеры
    print("\n6. Примеры рекомендаций:")
    print("-" * 40)
    for i, (_, rec) in enumerate(recommendations.head(5).iterrows()):
        print(f"Клиент {rec['client_code']} ({rec['name']}): {rec['product']} (уверенность: {rec['confidence']:.3f})")
    
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ УСПЕШНО СГЕНЕРИРОВАНЫ")
    print("=" * 60)
    print("Созданные файлы:")
    print("- recommendations.csv - итоговые рекомендации")
    print("\nДля генерации пуш-уведомлений запустите:")
    print("python run_push_generation.py")

if __name__ == "__main__":
    main()

