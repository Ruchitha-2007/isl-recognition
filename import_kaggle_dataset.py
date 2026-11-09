import os
import cv2
import glob
import numpy as np
import pandas as pd
from isl_engine import ISLLandmarkExtractor

def find_class_subfolders(root_dir):
    """
    Finds directories containing image files, handling potential nested folders
    like dataset/train/A or dataset/phrases/hello.
    """
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    
    # Check immediate subdirectories
    immediate_subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # If root_dir only contains 1 folder like 'data' or 'train', step inside
    if len(immediate_subdirs) == 1 and not any(glob.glob(os.path.join(immediate_subdirs[0], ext)) for ext in valid_exts):
        return find_class_subfolders(immediate_subdirs[0])
        
    class_folders = []
    for d in immediate_subdirs:
        # Check if this folder contains images directly
        has_images = any(glob.glob(os.path.join(d, ext)) for ext in valid_exts) or any(glob.glob(os.path.join(d, ext.upper())) for ext in valid_exts)
        if has_images:
            class_folders.append(d)
        else:
            # Check one level deeper (e.g. dataset/category/phrase)
            nested = [os.path.join(d, n) for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))]
            for nd in nested:
                if any(glob.glob(os.path.join(nd, ext)) for ext in valid_exts):
                    class_folders.append(nd)
                    
    return sorted(list(set(class_folders)))

def import_kaggle_dataset(kaggle_dir, output_csv="data/isl_landmarks.csv", samples_per_class=100):
    """
    Imports images from Kaggle ISL Alphabets or Phrases datasets.
    Extracts 126-feature MediaPipe normalized landmarks from images and merges them
    into data/isl_landmarks.csv.
    """
    if not os.path.exists(kaggle_dir):
        print(f"Error: Folder '{kaggle_dir}' does not exist.")
        return

    class_folders = find_class_subfolders(kaggle_dir)
    
    if not class_folders:
        print(f"No folders with images found inside '{kaggle_dir}'.")
        print("Please check that the folder contains images (.jpg, .png, etc.) inside class subfolders.")
        return

    extractor = ISLLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.5)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    print("=" * 70)
    print("      ISL DATASET (ALPHABETS / PHRASES) LANDMARK EXTRACTOR")
    print("=" * 70)
    print(f"Dataset path:       {kaggle_dir}")
    print(f"Found {len(class_folders)} classes.")
    print("Sample classes:     " + ", ".join([os.path.basename(f) for f in class_folders[:8]]) + "...")
    print(f"Max samples/class:  {samples_per_class}")
    print("Note: Signer face blurring does NOT affect hand tracking.")
    print("Extracting MediaPipe hand landmarks...")
    print("=" * 70)

    all_rows = []
    total_images_processed = 0
    total_landmarks_extracted = 0

    for folder_path in class_folders:
        raw_name = os.path.basename(folder_path)
        # Clean label name: e.g. 'thank_you' -> 'THANK YOU', 'good-morning' -> 'GOOD MORNING'
        label = raw_name.replace("_", " ").replace("-", " ").strip().upper()
        
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        image_files = list(set(image_files))
        if not image_files:
            continue
            
        # Select up to samples_per_class images (evenly spaced)
        if len(image_files) > samples_per_class:
            indices = np.linspace(0, len(image_files) - 1, samples_per_class, dtype=int)
            selected_files = [image_files[i] for i in indices]
        else:
            selected_files = image_files

        class_success = 0
        for img_path in selected_files:
            total_images_processed += 1
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            features, _, hands_detected = extractor.extract_landmarks(frame)
            
            # Keep only samples where MediaPipe successfully found hands
            if hands_detected > 0:
                row = list(features) + [label]
                all_rows.append(row)
                class_success += 1
                total_landmarks_extracted += 1

        print(f"Class [{label:18}]: Extracted {class_success:3d}/{len(selected_files):3d} valid hand frames.")

    extractor.close()

    # Save and Merge to CSV
    if all_rows:
        feature_cols = [f"f_{i}" for i in range(126)]
        new_df = pd.DataFrame(all_rows, columns=feature_cols + ["label"])
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            # Remove existing rows only for the classes we just imported to prevent duplicate bloat
            existing_df = existing_df[~existing_df['label'].isin(new_df['label'].unique())]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            
            unique_classes = combined_df['label'].unique()
            print("\n" + "=" * 70)
            print(f"MERGE SUCCESSFUL!")
            print(f"Total dataset size:    {len(combined_df)} samples")
            print(f"Total unique classes:  {len(unique_classes)} classes in '{output_csv}'")
            print("=" * 70)
        else:
            new_df.to_csv(output_csv, index=False)
            unique_classes = new_df['label'].unique()
            print(f"\nSUCCESS! Saved {len(new_df)} samples across {len(unique_classes)} classes to '{output_csv}'")

        print("\nNext step: Run the trainer to train your model with full evaluation metrics:")
        print("  python train_model.py\n")
    else:
        print("\nWarning: No hand landmarks could be extracted from these images.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter the path to your unzipped dataset folder: ").strip().strip('"')
    
    import_kaggle_dataset(target_dir)
