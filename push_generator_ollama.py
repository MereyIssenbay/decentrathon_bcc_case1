#!/usr/bin/env python3
"""
Генератор пуш-уведомлений через Ollama API
"""

import requests
import json
import pandas as pd
from typing import Dict, List
import time
import random
import os

class PushGeneratorOllama:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama3.2:3b"
        
        # Шаблоны для разных продуктов
        self.templates = {
            "Карта для путешествий": {
                "keywords": ["такси", "путешествия", "отели", "поездки"],
                "benefit": "4% кешбэк на путешествия и такси",
                "cta": "Откройте карту в приложении"
            },
            "Премиальная карта": {
                "keywords": ["высокий остаток", "премиум", "рестораны", "ювелирные"],
                "benefit": "до 4% кешбэка на все покупки и бесплатные снятия",
                "cta": "Подключите сейчас"
            },
            "Кредитная карта": {
                "keywords": ["топ-категории", "онлайн", "любимые категории"],
                "benefit": "до 10% в любимых категориях и на онлайн-сервисы",
                "cta": "Оформить карту"
            },
            "Обмен валют": {
                "keywords": ["валюта", "обмен", "курс"],
                "benefit": "выгодный обмен и авто-покупка по целевому курсу",
                "cta": "Настроить обмен"
            },
            "Кредит наличными": {
                "keywords": ["крупные траты", "финансирование"],
                "benefit": "кредит наличными с гибкими выплатами",
                "cta": "Узнать доступный лимит"
            },
            "Депозит Мультивалютный (KZT/USD/RUB/EUR)": {
                "keywords": ["свободные средства", "остаток", "копить"],
                "benefit": "удобно копить и получать вознаграждение",
                "cta": "Открыть вклад"
            },
            "Депозит Сберегательный (защита KDIF)": {
                "keywords": ["свободные средства", "остаток", "копить"],
                "benefit": "удобно копить и получать вознаграждение",
                "cta": "Открыть вклад"
            },
            "Депозит Накопительный": {
                "keywords": ["свободные средства", "остаток", "копить"],
                "benefit": "удобно копить и получать вознаграждение",
                "cta": "Открыть вклад"
            },
            "Инвестиции": {
                "keywords": ["свободные средства", "инвестиции"],
                "benefit": "низкий порог входа и без комиссий на старт",
                "cta": "Открыть счёт"
            },
            "Золотые слитки": {
                "keywords": ["свободные средства", "инвестиции", "золото"],
                "benefit": "надёжное сохранение стоимости и диверсификация",
                "cta": "Узнать подробнее"
            }
        }
    
    def check_ollama_connection(self) -> bool:
        """Проверяет подключение к Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_with_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Генерирует текст с помощью Ollama API"""
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 150
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    print(f"Ошибка API: {response.status_code}")
                    
            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def create_prompt(self, client_name: str, product: str, context: str) -> str:
        """Создает промпт для генерации пуш-уведомления"""
        
        template_info = self.templates.get(product, {})
        
        prompt = f"""Создай короткое пуш-уведомление для банковского продукта.

Клиент: {client_name}
Продукт: {product}
Контекст: {context}

Требования:
- Длина: 100-150 символов
- Обращение на "вы"
- Без КАПС
- Максимум 1 восклицательный знак
- Персонализированно
- Ясно и кратко
- Только на русском языке
- Без смешивания языков

Примеры:
"Рамазан, в августе вы сделали 12 поездок на такси на 27 400 ₸. С картой для путешествий вернули бы ≈1 100 ₸. Откройте карту в приложении."
"Алия, у вас высокий остаток на счету. Премиальная карта даст до 4% кешбэка на все покупки. Подключите сейчас."

Сгенерируй только текст уведомления:"""
        
        return prompt
    
    def generate_template_push(self, client_name: str, product: str, context: str) -> str:
        """Генерирует пуш-уведомление по шаблону"""
        
        template_info = self.templates.get(product, {})
        
        # Базовые шаблоны для каждого продукта
        templates = {
            "Карта для путешествий": [
                f"{client_name}, {context}. С картой для путешествий вернули бы кешбэк. {template_info.get('cta', 'Откройте карту в приложении')}.",
                f"{client_name}, у вас много поездок и такси. С тревел-картой часть расходов вернулась бы кешбэком. {template_info.get('cta', 'Откройте карту в приложении')}."
            ],
            "Премиальная карта": [
                f"{client_name}, {context}. Премиальная карта даст до 4% кешбэка на все покупки и бесплатные снятия. {template_info.get('cta', 'Подключите сейчас')}.",
                f"{client_name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк. {template_info.get('cta', 'Подключите сейчас')}."
            ],
            "Кредитная карта": [
                f"{client_name}, {context}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. {template_info.get('cta', 'Оформить карту')}.",
                f"{client_name}, {context}. Кредитная карта поможет сэкономить на покупках. {template_info.get('cta', 'Оформить карту')}."
            ],
            "Обмен валют": [
                f"{client_name}, вы часто меняете валюту. В приложении выгодный обмен и авто-покупка по целевому курсу. {template_info.get('cta', 'Настроить обмен')}.",
                f"{client_name}, {context}. В приложении выгодный обмен без комиссии. {template_info.get('cta', 'Настроить обмен')}."
            ],
            "Кредит наличными": [
                f"{client_name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. {template_info.get('cta', 'Узнать доступный лимит')}.",
                f"{client_name}, {context}. Кредит наличными поможет с финансированием. {template_info.get('cta', 'Узнать доступный лимит')}."
            ],
            "Депозит Мультивалютный (KZT/USD/RUB/EUR)": [
                f"{client_name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. {template_info.get('cta', 'Открыть вклад')}.",
                f"{client_name}, {context}. Мультивалютный депозит поможет сохранить и приумножить средства. {template_info.get('cta', 'Открыть вклад')}."
            ],
            "Депозит Сберегательный (защита KDIF)": [
                f"{client_name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. {template_info.get('cta', 'Открыть вклад')}.",
                f"{client_name}, {context}. Сберегательный депозит с защитой KDIF — надёжный способ накоплений. {template_info.get('cta', 'Открыть вклад')}."
            ],
            "Депозит Накопительный": [
                f"{client_name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. {template_info.get('cta', 'Открыть вклад')}.",
                f"{client_name}, {context}. Накопительный депозит поможет планомерно откладывать. {template_info.get('cta', 'Открыть вклад')}."
            ],
            "Инвестиции": [
                f"{client_name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. {template_info.get('cta', 'Открыть счёт')}.",
                f"{client_name}, {context}. Инвестиции помогут приумножить капитал. {template_info.get('cta', 'Открыть счёт')}."
            ],
            "Золотые слитки": [
                f"{client_name}, {context}. Золотые слитки — надёжное сохранение стоимости и диверсификация. {template_info.get('cta', 'Узнать подробнее')}.",
                f"{client_name}, у вас высокий остаток на счету. Золотые слитки помогут сохранить и приумножить капитал. {template_info.get('cta', 'Узнать подробнее')}."
            ]
        }
        
        product_templates = templates.get(product, [f"{client_name}, {context}. {template_info.get('benefit', 'Продукт поможет сэкономить')}. {template_info.get('cta', 'Узнать подробнее')}."])
        
        # Выбираем случайный шаблон
        return random.choice(product_templates)
    
    def validate_push_notification(self, text: str) -> bool:
        """Валидирует качество пуш-уведомления"""
        
        # Проверяем длину (очень мягкие требования)
        if len(text) < 50 or len(text) > 500:
            return False
        
        # Проверяем, что текст не пустой и содержит хотя бы одно предложение
        if len(text.strip()) < 10:
            return False
        
        # Проверяем отсутствие КАПС (только полный КАПС)
        if text.isupper():
            return False
        
        return True
    
    def generate_push_notification(self, client_name: str, product: str, context: str) -> str:
        """Генерирует пуш-уведомление для клиента"""
        
        # Проверяем подключение к Ollama
        if not self.check_ollama_connection():
            print("Ollama недоступен, пропускаем генерацию")
            return f"{client_name}, {product} - {context}"
        
        # Создаем промпт
        prompt = self.create_prompt(client_name, product, context)
        
        # Генерируем с помощью Ollama
        generated_text = self.generate_with_ollama(prompt)
        
        # Проверяем качество сгенерированного текста
        if generated_text and self.validate_push_notification(generated_text):
            return generated_text
        else:
            print(f"Сгенерированный текст не прошел валидацию: '{generated_text}'")
            print("Используем упрощенный вариант")
            return f"{client_name}, {product} - {context}"
    
    def generate_batch_notifications(self, recommendations_df: pd.DataFrame, 
                                   features_df: pd.DataFrame) -> pd.DataFrame:
        """Генерирует пуш-уведомления для списка рекомендаций"""
        
        print("Генерация пуш-уведомлений...")
        
        results = []
        
        for _, rec in recommendations_df.iterrows():
            client_code = rec['client_code']
            product = rec['product']
            
            # Получаем данные клиента
            client_data = features_df[features_df['client_code'] == client_code].iloc[0]
            client_name = client_data['name']
            
            # Создаем контекст клиента
            context = self.create_client_context(client_data)
            
            # Генерируем уведомление
            push_text = self.generate_push_notification(client_name, product, context)
            
            results.append({
                'client_code': client_code,
                'product': product,
                'push_notification': push_text
            })
        
        return pd.DataFrame(results)
    
    def create_client_context(self, client_data: pd.Series) -> str:
        """Создает контекст клиента для генерации пуш-уведомлений"""
        context_parts = []
        
        # Баланс и статус
        if client_data['avg_monthly_balance'] > 1000000:
            context_parts.append(f"высокий остаток на счету ({client_data['avg_monthly_balance']:,.0f} ₸)")
        elif client_data['avg_monthly_balance'] > 500000:
            context_parts.append(f"стабильный остаток на счету ({client_data['avg_monthly_balance']:,.0f} ₸)")
        
        # Траты на путешествия
        if client_data.get('travel_spent', 0) > 50000:
            context_parts.append(f"много трат на путешествия ({client_data['travel_spent']:,.0f} ₸)")
        if client_data.get('taxi_count', 0) > 10:
            context_parts.append(f"часто пользуетесь такси ({client_data['taxi_count']} поездок)")
        
        # Премиум траты
        if client_data.get('premium_spent', 0) > 100000:
            context_parts.append(f"траты в премиум категориях ({client_data['premium_spent']:,.0f} ₸)")
        
        # Онлайн активность
        if client_data.get('online_ratio', 0) > 0.3:
            context_parts.append("активно пользуетесь онлайн-сервисами")
        
        # FX операции
        if client_data.get('fx_count', 0) > 0:
            context_parts.append(f"часто меняете валюту ({client_data['fx_count']} операций)")
        
        return ", ".join(context_parts) if context_parts else "стабильное финансовое поведение"
    
    def save_notifications(self, notifications_df: pd.DataFrame, filename: str = "push_notifications.csv"):
        """Сохраняет пуш-уведомления в CSV"""
        notifications_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Пуш-уведомления сохранены в {filename}")
    
    def print_statistics(self, notifications_df: pd.DataFrame):
        """Выводит статистику пуш-уведомлений"""
        print("\n" + "="*50)
        print("СТАТИСТИКА ПУШ-УВЕДОМЛЕНИЙ")
        print("="*50)
        
        # Подсчет по продуктам
        product_counts = notifications_df['product'].value_counts()
        
        print("\nРаспределение уведомлений по продуктам:")
        for product, count in product_counts.items():
            percentage = (count / len(notifications_df)) * 100
            print(f"  {product}: {count} уведомлений ({percentage:.1f}%)")
        
        # Статистика по длине сообщений
        lengths = notifications_df['push_notification'].str.len()
        print(f"\nСредняя длина сообщения: {lengths.mean():.0f} символов")
        print(f"Минимальная длина: {lengths.min()} символов")
        print(f"Максимальная длина: {lengths.max()} символов")
        
        print(f"\nВсего уведомлений: {len(notifications_df)}")
        print("="*50)

if __name__ == "__main__":
    # Пример использования
    generator = PushGeneratorOllama()
    
    # Тестовые данные
    test_recommendations = pd.DataFrame([
        {'client_code': 1, 'product': 'Карта для путешествий'},
        {'client_code': 2, 'product': 'Премиальная карта'}
    ])
    
    test_features = pd.DataFrame([
        {'client_code': 1, 'name': 'Айгерим', 'avg_monthly_balance': 92643, 'travel_spent': 100000, 'taxi_count': 20},
        {'client_code': 2, 'name': 'Данияр', 'avg_monthly_balance': 1577073, 'premium_spent': 200000, 'fx_count': 5}
    ])
    
    # Генерируем уведомления
    notifications = generator.generate_batch_notifications(test_recommendations, test_features)
    
    # Сохраняем и выводим статистику
    generator.save_notifications(notifications)
    generator.print_statistics(notifications)
