import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def train_and_evaluate(data_path="data/isl_landmarks.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found! Run collect_data.py or generate_starter_data.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} samples with {df.shape[1] - 1} features.")
    
    # Filter out classes with fewer than 5 samples to guarantee stratified train/test integrity
    label_counts = df['label'].value_counts()
    valid_classes = label_counts[label_counts >= 5].index.tolist()
    dropped_classes = label_counts[label_counts < 5].index.tolist()
    
    if dropped_classes:
        print(f"\nNotice: Excluding {len(dropped_classes)} class(es) with < 5 samples: {dropped_classes}")
        print("Tip: If you want these signs, you can easily record them in 10s using 'python collect_data.py' (Option 3)!")
        df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
    
    # Feature columns vs target
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Encode target labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"\nActive Training Classes ({len(classes)}): {classes}")
    
    # Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Test Accuracy:          {acc * 100:.2f}%")
    print(f"Macro Precision:        {prec_macro * 100:.2f}%")
    print(f"Macro Recall:           {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:         {f1_macro * 100:.2f}%")
    print(f"Weighted Precision:     {prec_weighted * 100:.2f}%")
    print(f"Weighted Recall:        {rec_weighted * 100:.2f}%")
    print(f"Weighted F1-Score:      {f1_weighted * 100:.2f}%")
    print("="*50)
    
    rep = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Save Model and Encoders
    model_path = os.path.join(model_dir, "isl_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    report_path = os.path.join(model_dir, "evaluation_report.json")
    
    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    
    eval_summary = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "classes": classes,
        "classification_report": rep,
        "confusion_matrix": cm,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }
    
    with open(report_path, "w") as f:
        json.dump(eval_summary, f, indent=4)
        
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")
    print(f"Evaluation metrics saved to: {report_path}")
    return eval_summary

if __name__ == "__main__":
    train_and_evaluate()
