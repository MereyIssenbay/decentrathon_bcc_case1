
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    def __init__(self):
        self.products = [
            "Карта для путешествий",
            "Премиальная карта", 
            "Кредитная карта",
            "Обмен валют",
            "Кредит наличными",
            "Депозит Мультивалютный (KZT/USD/RUB/EUR)",
            "Депозит Сберегательный (защита KDIF)",
            "Депозит Накопительный",
            "Инвестиции",
            "Золотые слитки"
        ]
        
        # Разделение продуктов на категории
        self.mass_products = [
            "Карта для путешествий",
            "Кредитная карта", 
            "Обмен валют",
            "Кредит наличными",
            "Депозит Накопительный",
            "Инвестиции"
        ]
        
        self.premium_products = [
            "Премиальная карта",
            "Депозит Мультивалютный (KZT/USD/RUB/EUR)",
            "Депозит Сберегательный (защита KDIF)",
            "Золотые слитки"
        ]
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.rule_weights = {
            'travel': 0.3,
            'premium': 0.25,
            'credit': 0.2,
            'fx': 0.15,
            'deposit': 0.1
        }
        
    def load_data(self, data_dir: str = "data") -> pd.DataFrame:
        """Загружает и обрабатывает данные клиентов"""
        print("Загрузка данных...")
        
        # Загружаем профили клиентов
        clients_df = pd.read_csv(f"{data_dir}/clients.csv")
        
        # Загружаем транзакции и переводы для каждого клиента
        transactions = {}
        transfers = {}
        
        for i in range(1, 61):  # 60 клиентов
            try:
                trans_file = f"client_{i}_transactions_3m.csv"
                trans_path = f"{data_dir}/{trans_file}"
                
                if os.path.exists(trans_path):
                    transactions[i] = pd.read_csv(trans_path)
                    transactions[i]['date'] = pd.to_datetime(transactions[i]['date'])
                
                trans_file = f"client_{i}_transfers_3m.csv"
                trans_path = f"{data_dir}/{trans_file}"
                
                if os.path.exists(trans_path):
                    transfers[i] = pd.read_csv(trans_path)
                    transfers[i]['date'] = pd.to_datetime(transfers[i]['date'])
            except:
                continue
        
        # Создаем датасет с признаками
        all_features = []
        for client_id in range(1, 61):
            features = self.extract_features(client_id, clients_df, transactions, transfers)
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def extract_features(self, client_id: int, clients_df: pd.DataFrame, 
                        transactions: Dict, transfers: Dict) -> Dict:
        """Извлекает признаки для конкретного клиента"""
        
        # Базовые данные клиента
        client_data = clients_df[clients_df['client_code'] == client_id].iloc[0]
        
        # Определяем существующий продукт клиента из данных транзакций
        existing_product = ''
        if client_id in transactions:
            # Берем первый продукт из транзакций как существующий
            existing_product = transactions[client_id]['product'].iloc[0] if len(transactions[client_id]) > 0 else ''
        
        features = {
            'client_code': client_id,
            'name': client_data['name'],
            'age': client_data['age'],
            'status': client_data['status'],
            'city': client_data['city'],
            'avg_monthly_balance': client_data['avg_monthly_balance_KZT'],
            'existing_product': existing_product
        }
        
        # Обработка транзакций
        if client_id in transactions:
            trans_df = transactions[client_id]
            
            # Общие метрики трат
            features['total_spent'] = trans_df['amount'].sum()
            features['avg_transaction'] = trans_df['amount'].mean()
            features['transaction_count'] = len(trans_df)
            
            # Анализ по категориям
            category_spending = trans_df.groupby('category')['amount'].sum().to_dict()
            
            # Путешествия и такси
            travel_categories = ["Такси", "Отели", "Путешествия"]
            travel_spent = sum(category_spending.get(cat, 0) for cat in travel_categories)
            features['travel_spent'] = travel_spent
            features['travel_ratio'] = travel_spent / features['total_spent'] if features['total_spent'] > 0 else 0
            features['taxi_count'] = len(trans_df[trans_df['category'] == 'Такси'])
            
            # Премиум категории
            premium_categories = ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"]
            premium_spent = sum(category_spending.get(cat, 0) for cat in premium_categories)
            features['premium_spent'] = premium_spent
            features['premium_ratio'] = premium_spent / features['total_spent'] if features['total_spent'] > 0 else 0
            
            # Онлайн сервисы
            online_categories = ["Смотрим дома", "Играем дома", "Едим дома"]
            online_spent = sum(category_spending.get(cat, 0) for cat in online_categories)
            features['online_spent'] = online_spent
            features['online_ratio'] = online_spent / features['total_spent'] if features['total_spent'] > 0 else 0
            
        else:
            features.update({
                'total_spent': 0, 'avg_transaction': 0, 'transaction_count': 0,
                'travel_spent': 0, 'travel_ratio': 0, 'taxi_count': 0,
                'premium_spent': 0, 'premium_ratio': 0,
                'online_spent': 0, 'online_ratio': 0
            })
        
        # Обработка переводов
        if client_id in transfers:
            trans_df = transfers[client_id]
            
            # FX операции
            fx_operations = trans_df[trans_df['type'].isin(['fx_buy', 'fx_sell'])]
            features['fx_count'] = len(fx_operations)
            features['fx_amount'] = fx_operations['amount'].sum()
            
            # Снятия наличных
            atm_withdrawals = trans_df[trans_df['type'] == 'atm_withdrawal']
            features['atm_count'] = len(atm_withdrawals)
            features['atm_amount'] = atm_withdrawals['amount'].sum()
            
        else:
            features.update({
                'fx_count': 0, 'fx_amount': 0, 'atm_count': 0, 'atm_amount': 0
            })
        
        return features
    
    def rule_based_recommendation(self, features: Dict) -> Dict[str, float]:
        """Рекомендации на основе правил с учетом статуса и существующих продуктов"""
        scores = {}
        
        # Получаем существующий продукт клиента
        existing_product = features.get('existing_product', '')
        
        # Карта для путешествий
        travel_score = 0
        if existing_product != 'Карта для путешествий':
            if features['travel_ratio'] > 0.15:  # >15% трат на путешествия
                travel_score += 0.8
            if features['taxi_count'] > 15:  # Часто пользуется такси
                travel_score += 0.6
            if features['travel_spent'] > 80000:  # Много тратит на путешествия
                travel_score += 0.4
            # Бонус за статус
            if features['status'] in ['Зарплатный клиент', 'Стандартный клиент']:
                travel_score += 0.3
        scores['Карта для путешествий'] = travel_score
        
        # Премиальная карта
        premium_score = 0
        if existing_product != 'Премиальная карта':
            if features['avg_monthly_balance'] > 800000:  # Высокий баланс
                premium_score += 0.8
            if features['premium_ratio'] > 0.08:  # Тратит на премиум категории
                premium_score += 0.6
            if features['status'] in ['Премиальный клиент']:
                premium_score += 0.5
            elif features['status'] in ['Зарплатный клиент']:
                premium_score += 0.3
        scores['Премиальная карта'] = premium_score
        
        # Кредитная карта
        credit_score = 0
        if existing_product != 'Кредитная карта':
            if features['online_ratio'] > 0.25:  # Много онлайн трат
                credit_score += 0.8
            if features['age'] < 40:  # Молодой/средний возраст
                credit_score += 0.6
            if features['total_spent'] > 150000:  # Активно тратит
                credit_score += 0.4
            # Бонус за статус
            if features['status'] in ['Студент', 'Зарплатный клиент']:
                credit_score += 0.3
        scores['Кредитная карта'] = credit_score
        
        # Обмен валют
        fx_score = 0
        if existing_product != 'Обмен валют':
            if features['fx_count'] > 3:  # Часто меняет валюту
                fx_score += 0.8
            if features['fx_amount'] > 30000:  # Большие суммы обмена
                fx_score += 0.6
            # Бонус за статус
            if features['status'] in ['Премиальный клиент', 'Зарплатный клиент']:
                fx_score += 0.4
        scores['Обмен валют'] = fx_score
        
        # Кредит наличными
        loan_score = 0
        if existing_product != 'Кредит наличными':
            if features['total_spent'] > features['avg_monthly_balance'] * 0.7:  # Тратит >70% от баланса
                loan_score += 0.8
            if features['atm_count'] > 8:  # Часто снимает наличные
                loan_score += 0.6
            # Бонус за статус
            if features['status'] in ['Зарплатный клиент', 'Стандартный клиент']:
                loan_score += 0.3
        scores['Кредит наличными'] = loan_score
        
        # Депозит Мультивалютный
        deposit_multi_score = 0
        if existing_product != 'Депозит Мультивалютный (KZT/USD/RUB/EUR)':
            if features['avg_monthly_balance'] > 400000:  # Есть свободные средства
                deposit_multi_score += 0.8
            if features['fx_count'] > 0:  # Есть опыт с валютой
                deposit_multi_score += 0.4
            if features['status'] in ['Премиальный клиент', 'Зарплатный клиент']:
                deposit_multi_score += 0.5
            if features['age'] > 25:  # Взрослый клиент
                deposit_multi_score += 0.3
        scores['Депозит Мультивалютный (KZT/USD/RUB/EUR)'] = deposit_multi_score
        
        # Депозит Сберегательный
        deposit_save_score = 0
        if existing_product != 'Депозит Сберегательный (защита KDIF)':
            if features['avg_monthly_balance'] > 300000:  # Есть свободные средства
                deposit_save_score += 0.8
            if features['status'] in ['Премиальный клиент', 'Зарплатный клиент']:
                deposit_save_score += 0.5
            if features['age'] > 30:  # Взрослый клиент
                deposit_save_score += 0.4
            if features['age'] > 50:  # Пожилой клиент предпочитает сбережения
                deposit_save_score += 0.3
        scores['Депозит Сберегательный (защита KDIF)'] = deposit_save_score
        
        # Депозит Накопительный
        deposit_accum_score = 0
        if existing_product != 'Депозит Накопительный':
            if features['avg_monthly_balance'] > 350000:  # Есть свободные средства
                deposit_accum_score += 0.8
            if features['status'] in ['Зарплатный клиент', 'Стандартный клиент']:
                deposit_accum_score += 0.5
            if features['age'] > 25 and features['age'] < 45:  # Средний возраст
                deposit_accum_score += 0.4
        scores['Депозит Накопительный'] = deposit_accum_score
        
        # Инвестиции
        invest_score = 0
        if existing_product != 'Инвестиции':
            if features['avg_monthly_balance'] > 150000:  # Есть средства для инвестиций
                invest_score += 0.8
            if features['age'] < 55:  # Не слишком старый для инвестиций
                invest_score += 0.6
            if features['status'] in ['Премиальный клиент']:
                invest_score += 0.5
            elif features['status'] in ['Зарплатный клиент']:
                invest_score += 0.3
            if features['age'] < 35:  # Молодые более склонны к риску
                invest_score += 0.3
        scores['Инвестиции'] = invest_score
        
        # Золотые слитки - только для очень богатых клиентов
        gold_score = 0
        if existing_product != 'Золотые слитки':
            if features['avg_monthly_balance'] > 2000000:  # Очень высокий баланс
                gold_score += 0.8
            if features['status'] in ['Премиальный клиент']:
                gold_score += 0.6
            if features['age'] > 35:  # Взрослые клиенты
                gold_score += 0.4
            if features['premium_ratio'] > 0.15:  # Много премиум трат
                gold_score += 0.3
        scores['Золотые слитки'] = gold_score
        
        return scores
    
    def collaborative_filtering(self, features_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Коллаборативная фильтрация на основе похожих клиентов"""
        print("Выполнение коллаборативной фильтрации...")
        
        # Создаем матрицу признаков для поиска похожих клиентов
        feature_cols = ['age', 'avg_monthly_balance', 'total_spent', 'travel_ratio', 
                       'premium_ratio', 'online_ratio', 'fx_count']
        
        # Нормализуем признаки
        X = features_df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Вычисляем схожесть между клиентами
        similarities = {}
        for i, client1 in features_df.iterrows():
            similarities[client1['client_code']] = {}
            for j, client2 in features_df.iterrows():
                if i != j:
                    # Косинусное сходство
                    sim = np.dot(X_scaled[i], X_scaled[j]) / (
                        np.linalg.norm(X_scaled[i]) * np.linalg.norm(X_scaled[j]) + 1e-8
                    )
                    similarities[client1['client_code']][client2['client_code']] = sim
        
        # Рекомендации на основе похожих клиентов
        cf_scores = {}
        for client_id in features_df['client_code']:
            cf_scores[client_id] = {}
            
            # Находим топ-5 похожих клиентов
            similar_clients = sorted(similarities[client_id].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            
            # Агрегируем рекомендации похожих клиентов
            for product in self.products:
                score = 0
                total_weight = 0
                
                for similar_client_id, similarity in similar_clients:
                    if similarity > 0.3:  # Только достаточно похожие клиенты
                        # Получаем рекомендацию для похожего клиента
                        similar_features = features_df[features_df['client_code'] == similar_client_id].iloc[0]
                        rule_scores = self.rule_based_recommendation(similar_features.to_dict())
                        score += rule_scores.get(product, 0) * similarity
                        total_weight += similarity
                
                if total_weight > 0:
                    cf_scores[client_id][product] = score / total_weight
                else:
                    cf_scores[client_id][product] = 0
        
        return cf_scores
    
    def neural_network_recommendation(self, features_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Рекомендации на основе нейронной сети"""
        print("Обучение нейронной сети...")
        
        # Подготавливаем данные
        X, y = self.prepare_nn_data(features_df)
        
        # Создаем и обучаем модель
        model = self.build_neural_network(X.shape[1])
        
        # Обучаем модель
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train, 
                 validation_data=(X_test, y_test),
                 epochs=50, batch_size=16, verbose=0)
        
        # Получаем предсказания для всех клиентов
        predictions = model.predict(X, verbose=0)
        
        # Преобразуем в словарь
        nn_scores = {}
        for i, client_id in enumerate(features_df['client_code']):
            nn_scores[client_id] = {}
            for j, product in enumerate(self.products):
                nn_scores[client_id][product] = predictions[i][j]
        
        return nn_scores
    
    def prepare_nn_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для нейронной сети"""
        
        # Обрабатываем категориальные признаки
        categorical_features = ['status', 'city']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                features_df[feature] = self.label_encoders[feature].fit_transform(features_df[feature].astype(str))
            else:
                features_df[feature] = self.label_encoders[feature].transform(features_df[feature].astype(str))
        
        # Выбираем числовые признаки
        numerical_features = [
            'age', 'avg_monthly_balance', 'total_spent', 'travel_ratio', 
            'premium_ratio', 'online_ratio', 'fx_count', 'fx_amount'
        ]
        
        all_features = numerical_features + categorical_features
        
        # Создаем матрицу признаков
        X = features_df[all_features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Создаем целевые переменные на основе правил
        y = []
        for _, row in features_df.iterrows():
            rule_scores = self.rule_based_recommendation(row.to_dict())
            # Нормализуем scores
            max_score = max(rule_scores.values()) if max(rule_scores.values()) > 0 else 1
            normalized_scores = [rule_scores.get(product, 0) / max_score for product in self.products]
            y.append(normalized_scores)
        
        return X_scaled, np.array(y)
    
    def build_neural_network(self, input_dim: int) -> Sequential:
        """Строит нейронную сеть"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(self.products), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def hybrid_recommendation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Гибридная рекомендация с учетом категорий продуктов"""
        print("Выполнение гибридных рекомендаций...")
        
        # Получаем рекомендации от каждого метода
        rule_scores = {}
        for _, row in features_df.iterrows():
            rule_scores[row['client_code']] = self.rule_based_recommendation(row.to_dict())
        
        cf_scores = self.collaborative_filtering(features_df)
        nn_scores = self.neural_network_recommendation(features_df)
        
        # Объединяем рекомендации с весами
        final_recommendations = []
        
        # Счетчики для обеспечения разнообразия
        mass_product_counts = {product: 0 for product in self.mass_products}
        premium_product_counts = {product: 0 for product in self.premium_products}
        
        target_mass = len(features_df) * 0.7  # 70% массовых продуктов
        target_premium = len(features_df) * 0.3  # 30% премиум продуктов
        
        for _, row in features_df.iterrows():
            client_id = row['client_code']
            client_features = row.to_dict()
            
            # Взвешенная сумма всех методов
            combined_scores = {}
            for product in self.products:
                rule_score = rule_scores[client_id].get(product, 0)
                cf_score = cf_scores[client_id].get(product, 0)
                nn_score = nn_scores[client_id].get(product, 0)
                
                # Взвешенная сумма
                combined_score = (0.5 * rule_score + 0.3 * cf_score + 0.2 * nn_score)
                
                # Специальная логика для премиум продуктов
                if product in self.premium_products:
                    # Премиум продукты рекомендуем только при высокой уверенности
                    if combined_score > 0.8:  # Высокий порог для премиум продуктов
                        # Дополнительные бонусы для премиум клиентов
                        if client_features['status'] in ['Премиальный клиент']:
                            combined_score += 0.3
                        if client_features['avg_monthly_balance'] > 1000000:
                            combined_score += 0.2
                    else:
                        combined_score = 0  # Не рекомендуем премиум продукты при низкой уверенности
                
                # Бонусы за разнообразие
                if product in self.mass_products and mass_product_counts[product] < target_mass / len(self.mass_products):
                    combined_score += 0.1
                elif product in self.premium_products and premium_product_counts[product] < target_premium / len(self.premium_products):
                    combined_score += 0.2
                
                combined_scores[product] = combined_score
            
            # Находим лучший продукт
            best_product = max(combined_scores, key=combined_scores.get)
            
            # Обновляем счетчики
            if best_product in self.mass_products:
                mass_product_counts[best_product] += 1
            else:
                premium_product_counts[best_product] += 1
            
            final_recommendations.append({
                'client_code': client_id,
                'name': row['name'],
                'product': best_product,
                'confidence': combined_scores[best_product]
            })
        
        return pd.DataFrame(final_recommendations)
    
    def save_recommendations(self, recommendations_df: pd.DataFrame, filename: str = "recommendations.csv"):
        """Сохраняет рекомендации в CSV"""
        recommendations_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Рекомендации сохранены в {filename}")
    
    def print_statistics(self, recommendations_df: pd.DataFrame):
        """Выводит статистику рекомендаций"""
        print("\n" + "="*50)
        print("СТАТИСТИКА РЕКОМЕНДАЦИЙ")
        print("="*50)
        
        # Подсчет по продуктам
        product_counts = recommendations_df['product'].value_counts()
        
        print("\nРаспределение рекомендаций по продуктам:")
        for product, count in product_counts.items():
            percentage = (count / len(recommendations_df)) * 100
            print(f"  {product}: {count} клиентов ({percentage:.1f}%)")
        
        # Статистика по уверенности
        print(f"\nСредняя уверенность: {recommendations_df['confidence'].mean():.3f}")
        print(f"Максимальная уверенность: {recommendations_df['confidence'].max():.3f}")
        print(f"Минимальная уверенность: {recommendations_df['confidence'].min():.3f}")
        
        print(f"\nВсего рекомендаций: {len(recommendations_df)}")
        print("="*50)

if __name__ == "__main__":
    import os
    
    # Создаем и запускаем рекомендательную систему
    recommender = HybridRecommender()
    
    # Загружаем данные
    features_df = recommender.load_data()
    
    # Получаем рекомендации
    recommendations = recommender.hybrid_recommendation(features_df)
    
    # Сохраняем и выводим статистику
    recommender.save_recommendations(recommendations)
    recommender.print_statistics(recommendations)
