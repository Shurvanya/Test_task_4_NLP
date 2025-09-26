# ==================== usage.py ====================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== ВАЖНО: ИМПОРТ КЛАССОВ ИЗ ОРИГИНАЛЬНОГО ФАЙЛА ====================

# Вариант 1: Если оригинальный файл называется ecg_llm_ml_classifier.py
# from ecg_llm_ml_classifier import ECGClassifierLLM_ML, ECGDataHandler, LLMProcessor

# Вариант 2: Воссоздаем необходимые классы
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    DATA_PATH = 'ecg_data.csv'
    MISSING_VALUE = 29999
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    USE_LOCAL_LLM = True
    BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
    SENTENCE_MODEL = "all-MiniLM-L6-v2"
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

# ==================== ВОССОЗДАНИЕ КЛАССОВ ====================

class ECGDataHandler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def engineer_features(self, df):
        """Создание новых признаков из ЭКГ данных"""
        features = pd.DataFrame()
        
        # Базовые интервалы
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
        
        # QTc
        features['qtc_bazett'] = features['qt_interval'] / np.sqrt(features['rr_interval'] / 1000)
        
        # Соотношения
        features['pr_rr_ratio'] = features['pr_interval'] / features['rr_interval']
        features['qrs_qt_ratio'] = features['qrs_duration'] / features['qt_interval']
        
        # Индикаторы аномалий
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

class LLMProcessor:
    def __init__(self):
        print(f"Инициализация LLM на {Config.DEVICE}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BIOBERT_MODEL)
        self.model = AutoModel.from_pretrained(Config.BIOBERT_MODEL).to(Config.DEVICE)
        self.model.eval()
        
        self.sentence_transformer = SentenceTransformer(Config.SENTENCE_MODEL)
        
    def create_medical_report(self, row):
        """Создание медицинского текстового отчета из данных ЭКГ"""
        def format_value(val, suffix=""):
            if pd.isna(val):
                return "not measured"
            return f"{val:.0f}{suffix}"
        
        hr = 60000 / row['rr_interval'] if not pd.isna(row['rr_interval']) and row['rr_interval'] > 0 else np.nan
        pr = row['pr_interval'] if not pd.isna(row['pr_interval']) else np.nan
        qrs = row['qrs_duration'] if not pd.isna(row['qrs_duration']) else np.nan
        qt = row['qt_interval'] if not pd.isna(row['qt_interval']) else np.nan
        
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
        
        findings = []
        
        if not pd.isna(hr):
            if hr < 60:
                findings.append("Bradycardia (HR < 60 bpm)")
            elif hr > 100:
                findings.append("Tachycardia (HR > 100 bpm)")
            else:
                findings.append("Normal heart rate")
        
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
        
        if not pd.isna(row['qrs_axis']):
            axis = row['qrs_axis']
            if -30 <= axis <= 90:
                findings.append("Normal QRS axis")
            elif axis < -30:
                findings.append("Left axis deviation")
            elif axis > 90:
                findings.append("Right axis deviation")
        
        if findings:
            report += "\n".join(f"- {finding}" for finding in findings)
        else:
            report += "- Insufficient data for complete interpretation"
            
        critical_findings = [f for f in findings if any(word in f.lower() 
                           for word in ['block', 'prolonged', 'risk', 'tachy', 'brady'])]
        
        if critical_findings:
            report += f"\n\nCONCLUSION: Abnormal ECG - {len(critical_findings)} significant finding(s)"
        elif len(findings) > 0 and "Normal" in str(findings):
            report += "\n\nCONCLUSION: ECG within normal limits"
        else:
            report += "\n\nCONCLUSION: Incomplete ECG data"
            
        return report
    
    def get_sentence_embeddings(self, texts):
        """Быстрое получение эмбеддингов через sentence-transformers"""
        return self.sentence_transformer.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        )

class ECGClassifierLLM_ML:
    def __init__(self):
        self.data_handler = ECGDataHandler()
        self.llm_processor = LLMProcessor()
        
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
        
        self.weights = None
        
    def prepare_features(self, df):
        """Подготовка всех типов признаков"""
        print("Создание инженерных признаков...")
        engineered_features = self.data_handler.engineer_features(df)
        
        print("Генерация медицинских отчетов через LLM...")
        medical_reports = []
        for idx, row in engineered_features.iterrows():
            report = self.llm_processor.create_medical_report(row)
            medical_reports.append(report)
        
        print("Извлечение эмбеддингов из текстов...")
        text_embeddings = self.llm_processor.get_sentence_embeddings(medical_reports)
        
        engineered_features_imputed = pd.DataFrame(
            self.data_handler.imputer.fit_transform(engineered_features),
            columns=engineered_features.columns
        )
        
        engineered_features_scaled = pd.DataFrame(
            self.data_handler.scaler.fit_transform(engineered_features_imputed),
            columns=engineered_features.columns
        )
        
        embedding_df = pd.DataFrame(
            text_embeddings,
            columns=[f'embed_{i}' for i in range(text_embeddings.shape[1])]
        )
        
        all_features = pd.concat([
            engineered_features_scaled.reset_index(drop=True),
            embedding_df
        ], axis=1)
        
        return all_features, medical_reports
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        predictions = {}
        
        for name, model in self.ml_models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X)[:, 1]
        
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

# ==================== КЛАСС ДЛЯ ИСПОЛЬЗОВАНИЯ МОДЕЛИ ====================

class ECGPredictor:
    def __init__(self, model_path='ecg_classifier_llm_ml.pkl'):
        """Загрузка сохраненной модели"""
        print(f"Загрузка модели из {model_path}...")
        
        # Проверка существования файла
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        print("Модель успешно загружена!")
        
    def predict_single(self, ecg_data):
        """Предсказание для одной записи ЭКГ"""
        if isinstance(ecg_data, dict):
            df = pd.DataFrame([ecg_data])
        elif isinstance(ecg_data, pd.Series):
            df = pd.DataFrame([ecg_data])
        else:
            df = ecg_data
        
        # Замена 29999 на NaN
        df = df.replace(29999, np.nan)
        
        X_features, medical_reports = self.classifier.prepare_features(df)
        
        probability = self.classifier.predict_proba(X_features)[0]
        prediction = int(probability > 0.5)
        
        return {
            'prediction': prediction,
            'prediction_label': 'Healthy' if prediction == 1 else 'Abnormal',
            'probability_healthy': float(probability),
            'probability_abnormal': float(1 - probability),
            'confidence': float(max(probability, 1 - probability)),
            'medical_report': medical_reports[0]
        }
    
    def predict_batch(self, csv_file=None, dataframe=None):
        """Предсказание для нескольких записей"""
        if csv_file:
            df = pd.read_csv(csv_file)
            if 'Healthy' in df.columns:
                df = df.drop('Healthy', axis=1)
        elif dataframe is not None:
            df = dataframe
        else:
            raise ValueError("Необходимо указать csv_file или dataframe")
        
        print(f"Обработка {len(df)} записей...")
        
        df = df.replace(29999, np.nan)
        
        X_features, medical_reports = self.classifier.prepare_features(df)
        
        probabilities = self.classifier.predict_proba(X_features)
        predictions = (probabilities > 0.5).astype(int)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['Healthy' if p == 1 else 'Abnormal' for p in predictions],
            'probability_healthy': probabilities,
            'probability_abnormal': 1 - probabilities,
            'confidence': np.maximum(probabilities, 1 - probabilities)
        })
        
        results['medical_report'] = medical_reports
        
        return results

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

if __name__ == "__main__":
    import sys
    import os
    
    try:
        # Параметры по умолчанию
        input_csv = 'input_ecg.csv'  # Входной файл
        output_csv = 'predictions.csv'  # Выходной файл
        model_path = 'ecg_classifier_llm_ml.pkl'  # Путь к модели
        
        # Если указаны аргументы командной строки
        if len(sys.argv) > 1:
            input_csv = sys.argv[1]
        if len(sys.argv) > 2:
            output_csv = sys.argv[2]
        if len(sys.argv) > 3:
            model_path = sys.argv[3]
        
        # Проверка существования входного файла
        if not os.path.exists(input_csv):
            print(f"Ошибка: Входной файл '{input_csv}' не найден!")
            print("\nИспользование:")
            print("python usage.py <входной_csv> <выходной_csv> [путь_к_модели]")
            print("\nПример:")
            print("python usage.py input_ecg.csv predictions.csv ecg_classifier_llm_ml.pkl")
            sys.exit(1)
        
        print(f"Загрузка данных из: {input_csv}")
        
        # Загружаем модель
        predictor = ECGPredictor(model_path)
        
        # Загружаем и обрабатываем данные
        input_df = pd.read_csv(input_csv)
        print(f"Загружено записей: {len(input_df)}")
        
        # Делаем предсказания для всего файла
        results = predictor.predict_batch(dataframe=input_df)
        
        # Создаем выходной DataFrame
        output_df = input_df.copy()
        
        # Если есть колонка 'Healthy', сохраняем её для сравнения
        if 'Healthy' in output_df.columns:
            output_df['Original_Label'] = output_df['Healthy']
            output_df = output_df.drop('Healthy', axis=1)
        
        # Добавляем результаты предсказаний
        output_df['Predicted_Healthy'] = results['prediction']
        output_df['Prediction_Label'] = results['prediction_label']
        output_df['Probability_Healthy'] = results['probability_healthy'].round(4)
        output_df['Probability_Abnormal'] = results['probability_abnormal'].round(4)
        output_df['Confidence'] = results['confidence'].round(4)
        
        # Опционально: добавляем медицинский отчет (можно закомментировать для компактности)
        # output_df['Medical_Report'] = results['medical_report']
        
        # Сохраняем результаты
        output_df.to_csv(output_csv, index=False)
        print(f"\nРезультаты сохранены в: {output_csv}")
        
        # Выводим статистику
        print("\n" + "="*60)
        print("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
        print("="*60)
        print(f"Всего записей: {len(results)}")
        print(f"Предсказано Healthy: {(results['prediction'] == 1).sum()} ({(results['prediction'] == 1).sum()/len(results)*100:.1f}%)")
        print(f"Предсказано Abnormal: {(results['prediction'] == 0).sum()} ({(results['prediction'] == 0).sum()/len(results)*100:.1f}%)")
        print(f"Средняя уверенность: {results['confidence'].mean():.1%}")
        
        # Если есть истинные метки, выводим точность
        if 'Original_Label' in output_df.columns:
            accuracy = (output_df['Original_Label'] == output_df['Predicted_Healthy']).mean()
            print(f"\nТочность на данных: {accuracy:.1%}")
        
        # Показываем первые несколько результатов
        print("\n" + "="*60)
        print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ (первые 5 записей)")
        print("="*60)
        display_cols = ['Predicted_Healthy', 'Prediction_Label', 'Probability_Healthy', 'Confidence']
        if 'Original_Label' in output_df.columns:
            display_cols = ['Original_Label'] + display_cols
        print(output_df[display_cols].head().to_string())
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nУбедитесь, что файл модели 'ecg_classifier_llm_ml.pkl' находится в текущей директории")
        print("или укажите правильный путь к модели третьим аргументом")
        
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
