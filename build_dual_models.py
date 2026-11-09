import os
import glob
import json
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from isl_engine import ISLLandmarkExtractor

def extract_features_from_folder(folder_path, samples_per_class=100):
    """Extracts invariant features from class subdirectories."""
    extractor = ISLLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.4)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    
    subdirs = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    all_rows = []
    
    print(f"\nProcessing '{folder_path}' ({len(subdirs)} classes)...")
    for class_name in subdirs:
        label = class_name.replace("_", " ").replace("-", " ").strip().upper()
        class_dir = os.path.join(folder_path, class_name)
        
        images = []
        for ext in valid_exts:
            images.extend(glob.glob(os.path.join(class_dir, ext)))
            images.extend(glob.glob(os.path.join(class_dir, ext.upper())))
            
        images = list(set(images))
        if not images:
            continue
            
        if len(images) > samples_per_class:
            indices = np.linspace(0, len(images) - 1, samples_per_class, dtype=int)
            selected = [images[i] for i in indices]
        else:
            selected = images
            
        success = 0
        for img_path in selected:
            img = cv2.imread(img_path)
            if img is None:
                continue
            feats, _, detected = extractor.extract_landmarks(img)
            if detected > 0:
                all_rows.append(list(feats) + [label])
                success += 1
                
        print(f"  [{label:18}]: {success:3d}/{len(selected):3d} valid landmark frames.")
        
    extractor.close()
    
    if all_rows:
        num_features = len(all_rows[0]) - 1
        cols = [f"f_{i}" for i in range(num_features)] + ["label"]
        return pd.DataFrame(all_rows, columns=cols)
    return pd.DataFrame()

def train_specific_model(df, model_save_name, report_save_name):
    """Trains a Random Forest classifier on the given DataFrame and saves report."""
    os.makedirs("models", exist_ok=True)
    
    # Filter classes with >= 5 samples
    counts = df['label'].value_counts()
    valid_classes = counts[counts >= 5].index.tolist()
    df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
    
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    
    print("\n" + "=" * 55)
    print(f"       MODEL METRICS: {model_save_name.upper()}")
    print("=" * 55)
    print(f"Classes ({len(classes)}): {classes[:8]}...")
    print(f"Test Accuracy:       {acc * 100:.2f}%")
    print(f"Macro Precision:     {prec_macro * 100:.2f}%")
    print(f"Macro Recall:        {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
    print("=" * 55)
    
    rep = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    joblib.dump(clf, os.path.join("models", f"{model_save_name}.pkl"))
    joblib.dump(le, os.path.join("models", f"{model_save_name}_encoder.pkl"))
    
    summary = {
        "model_type": model_save_name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "classes": classes,
        "classification_report": rep,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }
    
    with open(os.path.join("models", f"{report_save_name}.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary

def build_all():
    alphabets_dir = r"C:\Users\PURNANANDH\Downloads\dataset\isl_alphabets"
    phrases_dir = r"C:\Users\PURNANANDH\Downloads\dataset\isl_phrases"
    
    os.makedirs("data", exist_ok=True)
    
    # 1. Build Alphabets Model
    if os.path.exists(alphabets_dir):
        df_alpha = extract_features_from_folder(alphabets_dir, samples_per_class=60)
        if not df_alpha.empty:
            df_alpha.to_csv("data/isl_alphabets.csv", index=False)
            train_specific_model(df_alpha, "isl_alphabets_model", "alphabets_evaluation_report")
            
    # 2. Build Words/Phrases Model
    if os.path.exists(phrases_dir):
        df_words = extract_features_from_folder(phrases_dir, samples_per_class=40)
        if not df_words.empty:
            df_words.to_csv("data/isl_words.csv", index=False)
            train_specific_model(df_words, "isl_words_model", "words_evaluation_report")

if __name__ == "__main__":
    build_all()
