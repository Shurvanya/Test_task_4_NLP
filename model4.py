import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def create_ecg_text(row):
    return f"ECG reading: RR interval {row['rr_interval']}ms, P wave from {row['p_onset']}ms to {row['p_end']}ms, QRS complex from {row['qrs_onset']}ms to {row['qrs_end']}ms, T wave ends at {row['t_end']}ms, axes P:{row['p_axis']}° QRS:{row['qrs_axis']}° T:{row['t_axis']}°"

def prepare_data():
    df = pd.read_csv("ecg_data.csv")
    
    # Cleaning data
    df = df.replace([29999, -29999], np.nan).dropna()
    
    # Class balancing
    min_count = df['Healthy'].value_counts().min()
    df = df.groupby('Healthy').sample(n=min_count, random_state=42)
    
    # Text creation
    df['text'] = df.apply(create_ecg_text, axis=1)
    
    # train/val divide
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Healthy'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    return Dataset.from_pandas(train_df[['text', 'Healthy']]), Dataset.from_pandas(val_df[['text', 'Healthy']])

# Prepaiting data
train_dataset, val_dataset = prepare_data()

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True).rename_column("Healthy", "labels")
tokenized_val = val_dataset.map(tokenize_function, batched=True).rename_column("Healthy", "labels")

# Metrics
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

# Training parameters
training_args = TrainingArguments(
    output_dir="./ecg_classifier",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    seed=42
)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)

trainer.train()

# Model saving
model.save_pretrained("./ecg_classifier_final2")
tokenizer.save_pretrained("./ecg_classifier_final2")

# Prediction
def predict_ecg(rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis):
    text = f"ECG reading: RR interval {rr_interval}ms, P wave from {p_onset}ms to {p_end}ms, QRS complex from {qrs_onset}ms to {qrs_end}ms, T wave ends at {t_end}ms, axes P:{p_axis}° QRS:{qrs_axis}° T:{t_axis}°"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

# Final report
predictions = trainer.predict(tokenized_val)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Potentially anomalous', 'Healthy']))
