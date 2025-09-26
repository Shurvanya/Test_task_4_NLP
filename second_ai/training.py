# ==================== 2. ИМПОРТ БИБЛИОТЕК ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ==================== 3. КОНФИГУРАЦИЯ ====================
class Config:
    # Пути и параметры
    DATA_PATH = 'ecg_data.csv'
    MISSING_VALUE = 29999  # Значение для отсутствующих данных
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # LLM модели (выберите одну)
    USE_LOCAL_LLM = True  # True - локальная модель, False - API
    
    # Локальные модели
    BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
    SENTENCE_MODEL = "all-MiniLM-L6-v2"
    
    # Референсные значения (нормальные диапазоны)
    NORMAL_RANGES = {
        'rr_interval': (600, 1000),
        'pr_interval': (120, 200),
        'qrs_duration': (80, 120),
        'qt_interval': (350, 450),
        'p_axis': (-30, 75),
        'qrs_axis': (-30, 90),
        't_axis': (0, 90)
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 4. ОБРАБОТКА ДАННЫХ ====================
class ECGDataHandler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_clean_data(self, filepath):
        """Загрузка и очистка данных"""
        # Загрузка
        df = pd.read_csv(filepath)
        print(f"Загружено записей: {len(df)}")
        
        # Замена значений 29999 на NaN
        df = df.replace(Config.MISSING_VALUE, np.nan)
        
        # Статистика пропусков
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            print("\nПропущенные значения:")
            print(missing_stats[missing_stats > 0])
        
        print(f"\nРаспределение классов:")
        print(df['Healthy'].value_counts())
        print(f"Доля здоровых: {df['Healthy'].mean():.2%}")
        
        return df
    
    def engineer_features(self, df):
        """Создание новых признаков из ЭКГ данных"""
        features = pd.DataFrame()
        
        # Базовые интервалы (с обработкой NaN)
        features['pr_interval'] = df['qrs_onset'] - df['p_onset']
        features['qrs_duration'] = df['qrs_end'] - df['qrs_onset']
        features['qt_interval'] = df['t_end'] - df['qrs_onset']
        features['p_duration'] = df['p_end'] - df['p_onset']
        features['st_segment'] = df['t_end'] - df['qrs_end']
        
        # Исходные признаки
        features['rr_interval'] = df['rr_interval']
        features['p_axis'] = df['p_axis']
        features['qrs_axis'] = df['qrs_axis']
        features['t_axis'] = df['t_axis']
        
        # Частота сердечных сокращений
        features['heart_rate'] = 60000 / df['rr_interval'].replace(0, np.nan)
        
        # QTc (корректированный QT) - формула Bazett
        features['qtc_bazett'] = features['qt_interval'] / np.sqrt(features['rr_interval'] / 1000)
        
        # Соотношения
        features['pr_rr_ratio'] = features['pr_interval'] / features['rr_interval']
        features['qrs_qt_ratio'] = features['qrs_duration'] / features['qt_interval']
        
        # Индикаторы аномалий для каждого параметра
        for param, (low, high) in Config.NORMAL_RANGES.items():
            if param in features.columns:
                features[f'{param}_low'] = (features[param] < low).astype(int)
                features[f'{param}_high'] = (features[param] > high).astype(int)
                features[f'{param}_normal'] = ((features[param] >= low) & 
                                              (features[param] <= high)).astype(int)
        
        # Счетчик аномалий
        anomaly_cols = [col for col in features.columns if col.endswith('_low') or col.endswith('_high')]
        features['total_anomalies'] = features[anomaly_cols].sum(axis=1)
        
        # Индикатор отсутствующих данных
        features['has_missing'] = df.isnull().any(axis=1).astype(int)
        features['missing_count'] = df.isnull().sum(axis=1)
        
        return features

# ==================== 5. LLM КОМПОНЕНТ ====================
class LLMProcessor:
    def __init__(self):
        print(f"Инициализация LLM на {Config.DEVICE}...")
        
        # Загрузка моделей
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BIOBERT_MODEL)
        self.model = AutoModel.from_pretrained(Config.BIOBERT_MODEL).to(Config.DEVICE)
        self.model.eval()
        
        # Sentence transformer для быстрых эмбеддингов
        self.sentence_transformer = SentenceTransformer(Config.SENTENCE_MODEL)
        
    def create_medical_report(self, row):
        """Создание медицинского текстового отчета из данных ЭКГ"""
        # Обработка пропущенных значений
        def format_value(val, suffix=""):
            if pd.isna(val):
                return "not measured"
            return f"{val:.0f}{suffix}"
        
        # Расчет производных показателей
        hr = 60000 / row['rr_interval'] if not pd.isna(row['rr_interval']) and row['rr_interval'] > 0 else np.nan
        pr = row['pr_interval'] if not pd.isna(row['pr_interval']) else np.nan
        qrs = row['qrs_duration'] if not pd.isna(row['qrs_duration']) else np.nan
        qt = row['qt_interval'] if not pd.isna(row['qt_interval']) else np.nan
        
        # Создание структурированного отчета
        report = f"""ECG REPORT:
        
MEASUREMENTS:
- Heart Rate: {format_value(hr, ' bpm')}
- RR Interval: {format_value(row['rr_interval'], 'ms')}
- PR Interval: {format_value(pr, 'ms')}
- QRS Duration: {format_value(qrs, 'ms')}
- QT Interval: {format_value(qt, 'ms')}
- QTc (Bazett): {format_value(row.get('qtc_bazett', np.nan), 'ms')}

ELECTRICAL AXIS:
- P Wave Axis: {format_value(row['p_axis'], '°')}
- QRS Axis: {format_value(row['qrs_axis'], '°')}
- T Wave Axis: {format_value(row['t_axis'], '°')}

INTERPRETATION:
"""
        
        # Добавление интерпретации
        findings = []
        
        # Анализ ЧСС
        if not pd.isna(hr):
            if hr < 60:
                findings.append("Bradycardia (HR < 60 bpm)")
            elif hr > 100:
                findings.append("Tachycardia (HR > 100 bpm)")
            else:
                findings.append("Normal heart rate")
        
        # Анализ интервалов
        if not pd.isna(pr):
            if pr > 200:
                findings.append("First-degree AV block (PR > 200ms)")
            elif pr < 120:
                findings.append("Short PR interval")
                
        if not pd.isna(qrs):
            if qrs > 120:
                findings.append("Wide QRS complex - possible bundle branch block")
            elif qrs < 80:
                findings.append("Narrow QRS complex")
                
        if not pd.isna(row.get('qtc_bazett', np.nan)):
            qtc = row['qtc_bazett']
            if qtc > 450:
                findings.append("Prolonged QTc interval - risk of arrhythmia")
            elif qtc < 350:
                findings.append("Short QTc interval")
        
        # Анализ электрической оси
        if not pd.isna(row['qrs_axis']):
            axis = row['qrs_axis']
            if -30 <= axis <= 90:
                findings.append("Normal QRS axis")
            elif axis < -30:
                findings.append("Left axis deviation")
            elif axis > 90:
                findings.append("Right axis deviation")
        
        # Добавление результатов в отчет
        if findings:
            report += "\n".join(f"- {finding}" for finding in findings)
        else:
            report += "- Insufficient data for complete interpretation"
            
        # Итоговая оценка
        critical_findings = [f for f in findings if any(word in f.lower() 
                           for word in ['block', 'prolonged', 'risk', 'tachy', 'brady'])]
        
        if critical_findings:
            report += f"\n\nCONCLUSION: Abnormal ECG - {len(critical_findings)} significant finding(s)"
        elif len(findings) > 0 and "Normal" in str(findings):
            report += "\n\nCONCLUSION: ECG within normal limits"
        else:
            report += "\n\nCONCLUSION: Incomplete ECG data"
            
        return report
    
    def get_bert_embeddings(self, texts, batch_size=32):
        """Получение BERT эмбеддингов для текстов"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Токенизация
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(Config.DEVICE)
                
                # Получение эмбеддингов
                outputs = self.model(**encoded)
                
                # Используем pooled output или среднее по последнему слою
                if hasattr(outputs, 'pooler_output'):
                    batch_embeddings = outputs.pooler_output
                else:
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentence_embeddings(self, texts):
        """Быстрое получение эмбеддингов через sentence-transformers"""
        return self.sentence_transformer.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        )

# ==================== 6. ОБЪЕДИНЕННАЯ МОДЕЛЬ LLM + ML ====================
class ECGClassifierLLM_ML:
    def __init__(self):
        self.data_handler = ECGDataHandler()
        self.llm_processor = LLMProcessor()
        
        # ML модели
        self.ml_models = {
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=Config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1.5
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=Config.RANDOM_STATE
            )
        }
        
        # Веса для ансамбля
        self.weights = None
        
    def prepare_features(self, df):
        """Подготовка всех типов признаков"""
        # 1. Инженерные признаки
        print("Создание инженерных признаков...")
        engineered_features = self.data_handler.engineer_features(df)
        
        # 2. Создание медицинских отчетов
        print("Генерация медицинских отчетов через LLM...")
        medical_reports = []
        for idx, row in engineered_features.iterrows():
            report = self.llm_processor.create_medical_report(row)
            medical_reports.append(report)
        
        # 3. Получение LLM эмбеддингов
        print("Извлечение эмбеддингов из текстов...")
        
        # Используем sentence-transformers для скорости
        text_embeddings = self.llm_processor.get_sentence_embeddings(medical_reports)
        
        # Опционально: добавляем BERT эмбеддинги для лучшего качества
        # bert_embeddings = self.llm_processor.get_bert_embeddings(medical_reports)
        
        # 4. Обработка пропущенных значений в числовых признаках
        engineered_features_imputed = pd.DataFrame(
            self.data_handler.imputer.fit_transform(engineered_features),
            columns=engineered_features.columns
        )
        
        # 5. Масштабирование числовых признаков
        engineered_features_scaled = pd.DataFrame(
            self.data_handler.scaler.fit_transform(engineered_features_imputed),
            columns=engineered_features.columns
        )
        
        # 6. Объединение всех признаков
        # Конвертируем эмбеддинги в DataFrame
        embedding_df = pd.DataFrame(
            text_embeddings,
            columns=[f'embed_{i}' for i in range(text_embeddings.shape[1])]
        )
        
        # Объединяем все признаки
        all_features = pd.concat([
            engineered_features_scaled.reset_index(drop=True),
            embedding_df
        ], axis=1)
        
        return all_features, medical_reports
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Обучение всех моделей"""
        print("\nОбучение моделей:")
        
        train_predictions = {}
        val_predictions = {}
        
        for name, model in self.ml_models.items():
            print(f"  Обучение {name}...")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Предсказания вероятностей
            train_predictions[name] = model.predict_proba(X_train)[:, 1]
            
            if X_val is not None and y_val is not None:
                val_predictions[name] = model.predict_proba(X_val)[:, 1]
                val_score = roc_auc_score(y_val, val_predictions[name])
                print(f"    Validation AUC: {val_score:.4f}")
        
        # Оптимизация весов ансамбля на валидационной выборке
        if X_val is not None and y_val is not None:
            self.weights = self._optimize_weights(val_predictions, y_val)
            print(f"\nОптимальные веса ансамбля: {self.weights}")
        else:
            self.weights = {name: 1/len(self.ml_models) for name in self.ml_models}
        
        return train_predictions, val_predictions
    
    def _optimize_weights(self, predictions, y_true):
        """Оптимизация весов для ансамбля"""
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = sum(w * predictions[name] 
                              for w, name in zip(weights, predictions.keys()))
            return -roc_auc_score(y_true, weighted_pred)
        
        # Ограничения: веса должны суммироваться в 1
        constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(predictions))]
        
        # Начальные веса
        initial_weights = [1/len(predictions)] * len(predictions)
        
        # Оптимизация
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return dict(zip(predictions.keys(), result.x))
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        predictions = {}
        
        for name, model in self.ml_models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        # Взвешенное усреднение
        if self.weights:
            final_proba = sum(self.weights[name] * pred 
                            for name, pred in predictions.items())
        else:
            final_proba = np.mean(list(predictions.values()), axis=0)
        
        return final_proba
    
    def predict(self, X, threshold=0.5):
        """Бинарное предсказание"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

# ==================== 7. ОЦЕНКА МОДЕЛИ ====================
def evaluate_model(y_true, y_pred, y_proba=None):
    """Полная оценка модели"""
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print("="*50)
    
    # Базовые метрики
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Specificity': recall_score(1-y_true, 1-y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
    
    # Вывод метрик
    for metric, value in metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"TN: {cm[0,0]:4} | FP: {cm[0,1]:4}")
    print(f"FN: {cm[1,0]:4} | TP: {cm[1,1]:4}")
    
    # Дополнительная статистика
    print(f"\nВсего предсказаний: {len(y_true)}")
    print(f"Положительных (Healthy=1): {sum(y_true)} ({sum(y_true)/len(y_true)*100:.1f}%)")
    print(f"Отрицательных (Abnormal=0): {len(y_true)-sum(y_true)} ({(len(y_true)-sum(y_true))/len(y_true)*100:.1f}%)")
    
    return metrics

# ==================== 8. ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    print("="*60)
    print("ECG CLASSIFICATION SYSTEM (LLM + ML)")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print("="*60)
    
    # 1. Загрузка и подготовка данных
    print("\n1. ЗАГРУЗКА ДАННЫХ")
    print("-"*40)
    
    # Создание примера данных если файла нет
    try:
        data_handler = ECGDataHandler()
        df = data_handler.load_and_clean_data(Config.DATA_PATH)
    except FileNotFoundError:
        print("Файл не найден. Создание примера данных...")
        np.random.seed(Config.RANDOM_STATE)
        n_samples = 500
        
        # Генерация реалистичных данных
        healthy = np.random.binomial(1, 0.6, n_samples)
        
        data = {
            'Healthy': healthy,
            'rr_interval': np.where(healthy, 
                                   np.random.normal(800, 100, n_samples),
                                   np.random.normal(700, 150, n_samples)),
            'p_onset': np.where(np.random.random(n_samples) > 0.95, 
                              Config.MISSING_VALUE,
                              np.random.normal(40, 10, n_samples)),
            'p_end': np.where(np.random.random(n_samples) > 0.95,
                            Config.MISSING_VALUE,
                            np.random.normal(150, 15, n_samples)),
            'qrs_onset': np.random.normal(190, 20, n_samples),
            'qrs_end': np.where(healthy,
                               np.random.normal(280, 20, n_samples),
                               np.random.normal(310, 30, n_samples)),
            't_end': np.random.normal(550, 50, n_samples),
            'p_axis': np.where(np.random.random(n_samples) > 0.9,
                             Config.MISSING_VALUE,
                             np.where(healthy,
                                    np.random.normal(45, 20, n_samples),
                                    np.random.normal(60, 35, n_samples))),
            'qrs_axis': np.where(healthy,
                                np.random.normal(30, 25, n_samples),
                                np.random.normal(50, 40, n_samples)),
            't_axis': np.random.normal(45, 30, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(Config.DATA_PATH, index=False)
        print(f"Создан файл {Config.DATA_PATH} с {n_samples} записями")
        
        # Перезагрузка
        df = data_handler.load_and_clean_data(Config.DATA_PATH)
    
    # 2. Подготовка признаков
    print("\n2. ПОДГОТОВКА ПРИЗНАКОВ")
    print("-"*40)
    
    # Разделение на X и y
    X_raw = df.drop('Healthy', axis=1)
    y = df['Healthy'].values
    
    # Инициализация классификатора
    classifier = ECGClassifierLLM_ML()
    
    # Подготовка признаков (числовые + LLM эмбеддинги)
    X_features, medical_reports = classifier.prepare_features(X_raw)
    
    print(f"Размерность признаков: {X_features.shape}")
    print(f"Количество числовых признаков: {len([c for c in X_features.columns if not c.startswith('embed_')])}")
    print(f"Количество LLM эмбеддингов: {len([c for c in X_features.columns if c.startswith('embed_')])}")
    
    # 3. Разделение на обучающую и тестовую выборки
    print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ")
    print("-"*40)
    
    X_train, X_test, y_train, y_test, reports_train, reports_test = train_test_split(
        X_features, y, medical_reports,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"Обучающая выборка: {len(X_train)} записей")
    print(f"Тестовая выборка: {len(X_test)} записей")
    
    # 4. Обучение модели
    print("\n4. ОБУЧЕНИЕ МОДЕЛИ")
    print("-"*40)
    
    # Разделение обучающей выборки для валидации
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
        stratify=y_train
    )
    
    # Обучение
    train_preds, val_preds = classifier.train(X_tr, y_tr, X_val, y_val)
    
    # 5. Оценка на тестовой выборке
    print("\n5. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("-"*40)
    
    # Предсказания
    y_test_proba = classifier.predict_proba(X_test)
    y_test_pred = classifier.predict(X_test)
    
    # Оценка
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba)
    
    # 6. Анализ ошибок
    print("\n6. АНАЛИЗ ОШИБОК")
    print("-"*40)
    
    # Находим индексы ошибочных предсказаний
    errors = np.where(y_test != y_test_pred)[0]
    
    if len(errors) > 0:
        print(f"Количество ошибок: {len(errors)} ({len(errors)/len(y_test)*100:.1f}%)")
        
        # Анализ False Positives и False Negatives
        fp_indices = errors[y_test[errors] == 0]
        fn_indices = errors[y_test[errors] == 1]
        
        print(f"False Positives (предсказали Healthy, но Abnormal): {len(fp_indices)}")
        print(f"False Negatives (предсказали Abnormal, но Healthy): {len(fn_indices)}")
        
        # Пример ошибочной классификации
        if len(errors) > 0:
            print("\nПример ошибочной классификации:")
            error_idx = errors[0]
            print(f"Истинный класс: {y_test[error_idx]}")
            print(f"Предсказанный класс: {y_test_pred[error_idx]}")
            print(f"Вероятность класса 1: {y_test_proba[error_idx]:.3f}")
            print("\nМедицинский отчет для этой записи:")
            print(reports_test[error_idx][:500] + "...")
    
    # 7. Важность признаков
    print("\n7. ВАЖНОСТЬ ПРИЗНАКОВ")
    print("-"*40)
    
    # Получаем важность признаков из Random Forest
    rf_model = classifier.ml_models['rf']
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Топ-15 важных признаков
    print("\nТоп-15 наиболее важных признаков:")
    for idx, row in feature_importance.head(15).iterrows():
        feature_type = "LLM" if row['feature'].startswith('embed_') else "Engineered"
        print(f"{row['feature']:30} [{feature_type:10}]: {row['importance']:.4f}")
    
    # Соотношение важности LLM vs инженерных признаков
    llm_importance = feature_importance[feature_importance['feature'].str.startswith('embed_')]['importance'].sum()
    eng_importance = feature_importance[~feature_importance['feature'].str.startswith('embed_')]['importance'].sum()
    
    print(f"\nСуммарная важность:")
    print(f"  LLM признаки: {llm_importance:.3f} ({llm_importance/(llm_importance+eng_importance)*100:.1f}%)")
    print(f"  Инженерные признаки: {eng_importance:.3f} ({eng_importance/(llm_importance+eng_importance)*100:.1f}%)")
    
    # 8. Кросс-валидация
    print("\n8. КРОСС-ВАЛИДАЦИЯ (5-fold)")
    print("-"*40)
    
    from sklearn.model_selection import cross_validate
    
    # Используем только RandomForest для быстрой кросс-валидации
    cv_scores = cross_validate(
        classifier.ml_models['rf'],
        X_features, y,
        cv=5,
        scoring=['accuracy', 'roc_auc', 'f1'],
        n_jobs=-1
    )
    
    for metric in ['accuracy', 'roc_auc', 'f1']:
        scores = cv_scores[f'test_{metric}']
        print(f"{metric:10}: {scores.mean():.4f} (±{scores.std():.4f})")
    
    print("\n" + "="*60)
    print("КЛАССИФИКАЦИЯ ЗАВЕРШЕНА!")
    print("="*60)
    
    return classifier, test_metrics, medical_reports

# ==================== 9. ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    # Запуск основной программы
    try:
        classifier, metrics, reports = main()
        
        # Сохранение модели (опционально)
        import pickle
        with open('ecg_classifier_llm_ml.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        print("\nМодель сохранена в 'ecg_classifier_llm_ml.pkl'")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
