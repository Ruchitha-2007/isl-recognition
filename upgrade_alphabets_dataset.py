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

def build_expanded_alphabets_model(alphabets_dir=r"C:\Users\PURNANANDH\Downloads\dataset\isl_alphabets", samples_per_class=250):
    """
    Samples 250 images per alphabet class (across the 1,200 available)
    yielding ~6,500 training frames with full angle/scale coverage.
    """
    if not os.path.exists(alphabets_dir):
        print(f"Error: Directory '{alphabets_dir}' not found.")
        return

    extractor = ISLLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.35)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    subdirs = sorted([d for d in os.listdir(alphabets_dir) if os.path.isdir(os.path.join(alphabets_dir, d))])
    
    print("=" * 65)
    print("      UPGRADING ISL ALPHABETS DATASET (250 SAMPLES / CLASS)")
    print("=" * 65)
    print(f"Total alphabet classes: {len(subdirs)}")
    print(f"Target samples/class:   {samples_per_class}")
    print(f"Total target frames:    ~{len(subdirs) * samples_per_class}")
    print("=" * 65)
    
    all_rows = []
    
    for class_name in subdirs:
        label = class_name.strip().upper()
        class_dir = os.path.join(alphabets_dir, class_name)
        
        images = []
        for ext in valid_exts:
            images.extend(glob.glob(os.path.join(class_dir, ext)))
            images.extend(glob.glob(os.path.join(class_dir, ext.upper())))
            
        images = sorted(list(set(images)))
        if not images:
            continue
            
        # Evenly space out across all 1,200 images to capture maximum angle/hand variety
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
                
        print(f"Letter [{label:2}]: Extracted {success:3d}/{len(selected):3d} valid landmark frames.")
        
    extractor.close()
    
    if all_rows:
        num_features = len(all_rows[0]) - 1
        cols = [f"f_{i}" for i in range(num_features)] + ["label"]
        df = pd.DataFrame(all_rows, columns=cols)
        
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        csv_path = "data/isl_alphabets.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved expanded dataset ({len(df)} samples) to '{csv_path}'.")
        
        # Train and Evaluate Random Forest
        X = df.drop(columns=['label']).values
        y = df['label'].values
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = list(le.classes_)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("\nTraining Random Forest with 150 estimators on expanded dataset...")
        clf = RandomForestClassifier(
            n_estimators=150,
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
        print("    EXPANDED ALPHABETS MODEL EVALUATION REPORT")
        print("=" * 55)
        print(f"Total Samples:       {len(df)}")
        print(f"Test Accuracy:       {acc * 100:.2f}%")
        print(f"Macro Precision:     {prec_macro * 100:.2f}%")
        print(f"Macro Recall:        {rec_macro * 100:.2f}%")
        print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
        print("=" * 55)
        
        rep = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
        
        model_path = "models/isl_alphabets_model.pkl"
        encoder_path = "models/isl_alphabets_model_encoder.pkl"
        report_path = "models/alphabets_evaluation_report.json"
        
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        
        summary = {
            "model_type": "isl_alphabets_model",
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
        
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Updated model saved to '{model_path}'.")
        print(f"Updated report saved to '{report_path}'.")
        return summary

if __name__ == "__main__":
    build_expanded_alphabets_model()
