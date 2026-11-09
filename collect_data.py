import os
import cv2
import time
import numpy as np
import pandas as pd
from isl_engine import ISLLandmarkExtractor

# 1. STARTER PACK (Curated Signs & Common Words - Perfect for 1-2 day testing)
STARTER_20_SIGNS = [
    # Core High-frequency letters
    ("A", "Left palm flat facing up; Right index finger touches Left thumb tip.", "2 Hands"),
    ("B", "Left palm flat facing up; Right index touches Left index finger tip.", "2 Hands"),
    ("C", "Right hand: Curved into a 'C' arc shape facing left.", "1 Hand"),
    ("D", "Left index finger held upright; Right hand forms arc touching tip & base of Left index.", "2 Hands"),
    ("E", "Left palm flat facing up; Right index touches Left middle finger tip.", "2 Hands"),
    ("F", "Left index & middle horizontal; Right index & middle cross over them forming 'F'.", "2 Hands"),
    ("G", "Both hands in fists with thumbs resting over index, stacked vertically.", "2 Hands"),
    ("H", "Left palm flat facing right; Right hand sweeps horizontal across left palm.", "2 Hands"),
    ("I", "Left palm flat facing up; Right index touches Left ring finger tip.", "2 Hands"),
    ("L", "Right hand: Thumb and index finger extended at 90 degrees forming an 'L'.", "1 Hand"),
    ("M", "Left open palm; Right hand puts 3 fingers (index, middle, ring) down on left palm.", "2 Hands"),
    ("N", "Left open palm; Right hand puts 2 fingers (index, middle) down on left palm.", "2 Hands"),
    ("O", "Left palm flat facing up; Right index touches Left pinky finger tip.", "2 Hands"),
    ("P", "Left index held upright; Right index & thumb form circle touching tip of Left index.", "2 Hands"),
    ("T", "Left index held horizontal; Right index touches bottom of left index forming 'T'.", "2 Hands"),
    ("V", "Right hand: Index and middle fingers extended apart forming a 'V' peace sign.", "1 Hand"),
    
    # Common Words & Expressions
    ("HELLO", "Right hand open palm raised to temple in a wave/salute.", "1 Hand"),
    ("NAMASTE", "Both palms joined together in front of chest in traditional greeting.", "2 Hands"),
    ("YES", "Right fist nodding up and down at the wrist like a head nodding.", "1 Hand"),
    ("NO", "Right index, middle finger and thumb snapping together like a bird's beak.", "1 Hand"),
    ("I LOVE YOU", "Right hand: Thumb, Index, and Pinky fingers extended upright (middle & ring folded).", "1 Hand"),
    ("THANK YOU", "Right fingertips touch chin then move forward towards camera with open palm.", "1 Hand")
]

# 2. EXTENDED PACK (Full A-Z, Numbers 0-9 & Key Conversational Words)
EXTENDED_50_SIGNS = [
    # Full A-Z (26 Signs)
    ("A", "Left palm flat facing up; Right index finger touches Left thumb tip.", "2 Hands"),
    ("B", "Left palm flat facing up; Right index touches Left index finger tip.", "2 Hands"),
    ("C", "Right hand: Curved into a 'C' arc shape facing left.", "1 Hand"),
    ("D", "Left index finger held upright; Right hand forms arc touching tip and base of Left index.", "2 Hands"),
    ("E", "Left palm flat facing up; Right index touches Left middle finger tip.", "2 Hands"),
    ("F", "Left index & middle horizontal; Right index & middle cross over them forming 'F'.", "2 Hands"),
    ("G", "Both hands in fists with thumbs over index, stacked vertically.", "2 Hands"),
    ("H", "Left palm flat facing right; Right hand sweeps horizontal across left palm.", "2 Hands"),
    ("I", "Left palm flat facing up; Right index touches Left ring finger tip.", "2 Hands"),
    ("J", "Left palm flat; Right index traces downward hook across left palm.", "2 Hands"),
    ("K", "Left index pointed up; Right index and thumb form a V touching left index joint.", "2 Hands"),
    ("L", "Right hand: Thumb and index finger extended at 90 degrees forming an 'L'.", "1 Hand"),
    ("M", "Left open palm; Right hand puts 3 fingers (index, middle, ring) down on left palm.", "2 Hands"),
    ("N", "Left open palm; Right hand puts 2 fingers (index, middle) down on left palm.", "2 Hands"),
    ("O", "Left palm flat facing up; Right index touches Left pinky finger tip.", "2 Hands"),
    ("P", "Left index held upright; Right index & thumb form circle touching tip of Left index.", "2 Hands"),
    ("Q", "Left hand makes circle; Right index finger tucks under circle forming hook.", "2 Hands"),
    ("R", "Left palm open; Right index finger curled and hooked over left palm.", "2 Hands"),
    ("S", "Both hands: Left pinky finger hooked with Right pinky finger.", "2 Hands"),
    ("T", "Left index held horizontal; Right index touches bottom of left index forming 'T'.", "2 Hands"),
    ("U", "Left palm flat facing right; Right index traces cup-shaped 'U' on palm.", "2 Hands"),
    ("V", "Right hand: Index and middle fingers extended apart forming a 'V' peace sign.", "1 Hand"),
    ("W", "Both hands: Interlocking upright fingers forming 'W' peaks.", "2 Hands"),
    ("X", "Both hands: Both index fingers crossed over each other forming an 'X'.", "2 Hands"),
    ("Y", "Left index extended upright; Right V-sign placed on top of Left index.", "2 Hands"),
    ("Z", "Left palm held vertical facing side; Right index touches left palm diagonally.", "2 Hands"),
    
    # Numbers 0-9 (10 Signs)
    ("0", "Right hand forms an 'O' ring with all fingers touching thumb.", "1 Hand"),
    ("1", "Right hand: Index finger extended straight up.", "1 Hand"),
    ("2", "Right hand: Index and middle fingers extended straight up.", "1 Hand"),
    ("3", "Right hand: Thumb, index, and middle fingers extended.", "1 Hand"),
    ("4", "Right hand: 4 fingers extended, thumb folded across palm.", "1 Hand"),
    ("5", "Right hand: Open palm with all 5 fingers spread out.", "1 Hand"),
    ("6", "Both hands: Left hand shows 5 fingers, Right hand points 1 finger.", "2 Hands"),
    ("7", "Both hands: Left hand shows 5 fingers, Right hand shows 2 fingers.", "2 Hands"),
    ("8", "Both hands: Left hand shows 5 fingers, Right hand shows 3 fingers.", "2 Hands"),
    ("9", "Both hands: Left hand shows 5 fingers, Right hand shows 4 fingers.", "2 Hands"),
    
    # Conversational Words (12 Signs)
    ("HELLO", "Right hand open palm raised to temple in a wave/salute.", "1 Hand"),
    ("NAMASTE", "Both palms joined together in front of chest in traditional greeting.", "2 Hands"),
    ("PLEASE", "Right hand flat palm rubbed gently over chest in clockwise circle.", "1 Hand"),
    ("THANK YOU", "Right fingertips touch chin then move forward towards camera with open palm.", "1 Hand"),
    ("YES", "Right fist nodding up and down like a head nodding.", "1 Hand"),
    ("NO", "Right index, middle finger and thumb snapping together like a beak.", "1 Hand"),
    ("HELP", "Left flat palm facing up; Right thumbs-up placed firmly on left palm.", "2 Hands"),
    ("GOOD", "Right hand thumbs up gesture held in front of chest.", "1 Hand"),
    ("BAD", "Right hand thumbs down gesture held in front of chest.", "1 Hand"),
    ("WATER", "Right hand forms 'W' (index, middle, ring up) tapping chin twice.", "1 Hand"),
    ("FOOD", "Right fingers pinched together tapping lips twice.", "1 Hand"),
    ("FRIEND", "Both index fingers hooked together interlocking both hands.", "2 Hands"),
    ("I LOVE YOU", "Right hand: Thumb, Index, and Pinky fingers extended upright (middle & ring folded).", "1 Hand")
]

def draw_text_with_bg(img, text, pos, font_scale=0.6, font_color=(255, 255, 255), bg_color=(20, 20, 20), thickness=1, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

def select_sign_pack():
    print("\n" + "=" * 65)
    print("      INDIAN SIGN LANGUAGE (ISL) RECORDER - PACK SELECTION")
    print("=" * 65)
    print("  [1] Starter Set (22 Signs: 16 letters + HELLO, NAMASTE, YES, NO, I LOVE YOU, etc.)")
    print("      -> Recommended to test accuracy & get started quickly!")
    print("  [2] Extended Set (49 Signs: Full A-Z, Numbers 0-9, and Common Words)")
    print("      -> Choose this when you're ready to expand your dataset.")
    print("  [3] Custom Sign (Record any specific word or letter of your choice)")
    print("=" * 65)
    
    choice = input("Enter your choice (1, 2, or 3) [Default: 1]: ").strip()
    
    if choice == "2":
        return EXTENDED_50_SIGNS, "Extended Pack (50 Signs)"
    elif choice == "3":
        custom_name = input("Enter the name of your custom sign/word (e.g. INDIA): ").strip().upper()
        if not custom_name:
            custom_name = "CUSTOM_SIGN"
        desc = input(f"Brief description of how to do [{custom_name}]: ").strip()
        if not desc:
            desc = f"Hand gesture for {custom_name}."
        h_type = input("Uses 1 Hand or 2 Hands? (1 or 2) [Default: 2]: ").strip()
        h_type_str = "1 Hand" if h_type == "1" else "2 Hands"
        return [(custom_name, desc, h_type_str)], f"Custom Sign: [{custom_name}]"
    else:
        return STARTER_20_SIGNS, "Starter Pack (20 Signs)"

def run_data_collection(record_duration=5.0, output_csv="data/isl_landmarks.csv"):
    signs_to_record, pack_title = select_sign_pack()
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    extractor = ISLLandmarkExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam. Please check your camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_idx = 0
    total_signs = len(signs_to_record)
    all_data = []

    # Check already collected samples
    existing_samples_count = {}
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            existing_samples_count = df_existing['label'].value_counts().to_dict()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(f"  ACTIVE RECORDING: {pack_title} ({total_signs} Total Signs)")
    print("=" * 70)
    print(f"Each recording runs for exactly {record_duration} seconds.")
    print("Labeling is 100% AUTOMATIC. The active sign is shown before recording.")
    print("\nControls:")
    print("  [SPACE] or [s] : Start 5-second recording for current sign")
    print("  [n]            : Skip to Next sign")
    print("  [b]            : Go Back to Previous sign")
    print("  [r]            : Re-record current sign")
    print("  [p]            : Pause / Take a break")
    print("  [q]            : Save progress and Quit anytime")
    print("=" * 70)

    state = "PRACTICE"
    countdown_start = 0
    record_start = 0
    current_sign_samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        features, annotated_frame, hands_detected = extractor.extract_landmarks(frame)

        sign_name, sign_desc, hand_type = signs_to_record[current_idx]
        prev_count = existing_samples_count.get(sign_name, 0)

        # Top Information Banner
        cv2.rectangle(annotated_frame, (0, 0), (w, 140), (15, 23, 42), -1)
        cv2.line(annotated_frame, (0, 140), (w, 140), (56, 189, 248), 2)

        # Sign Title & Automatic Label Name
        cv2.putText(annotated_frame, f"SIGN LABEL: [{sign_name}] ({current_idx + 1}/{total_signs})",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (56, 189, 248), 3, cv2.LINE_AA)
        
        # Mode Badge
        draw_text_with_bg(annotated_frame, f"Mode: {hand_type}", (w - 230, 45),
                          font_scale=0.7, font_color=(199, 210, 254), bg_color=(49, 46, 129), thickness=2)

        # Instructions
        cv2.putText(annotated_frame, f"How to do: {sign_desc}",
                    (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (241, 245, 249), 1, cv2.LINE_AA)
        
        # Detection Status
        hands_color = (16, 185, 129) if hands_detected > 0 else (239, 68, 68)
        cv2.putText(annotated_frame, f"Hands in frame: {hands_detected}/2",
                    (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hands_color, 2, cv2.LINE_AA)

        if prev_count > 0:
            cv2.putText(annotated_frame, f"Existing Samples in CSV: {prev_count}",
                        (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2, cv2.LINE_AA)

        # Bottom Control Panel
        cv2.rectangle(annotated_frame, (0, h - 60), (w, h), (15, 23, 42), -1)
        cv2.line(annotated_frame, (0, h - 60), (w, h - 60), (30, 41, 59), 1)

        # State Handling
        if state == "PRACTICE":
            cv2.putText(annotated_frame, f"[PRACTICE MODE] Auto-Label: [{sign_name}]. Press [SPACE] to Record 5s | [p] Pause | [n] Next | [q] Quit",
                        (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (203, 213, 225), 1, cv2.LINE_AA)

        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remaining = 3.0 - elapsed
            if remaining <= 0:
                state = "RECORDING"
                record_start = time.time()
                current_sign_samples = []
            else:
                num = int(np.ceil(remaining))
                cv2.circle(annotated_frame, (w // 2, h // 2), 90, (15, 23, 42), -1)
                cv2.circle(annotated_frame, (w // 2, h // 2), 90, (56, 189, 248), 4)
                cv2.putText(annotated_frame, str(num), (w // 2 - 25, h // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.2, (56, 189, 248), 6, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Hold Sign [{sign_name}] steadily!",
                            (w // 2 - 190, h // 2 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        elif state == "RECORDING":
            elapsed = time.time() - record_start
            remaining = record_duration - elapsed

            if hands_detected > 0:
                # Automatically tagged with the active sign_name!
                current_sign_samples.append(list(features) + [sign_name])

            if remaining <= 0:
                all_data.extend(current_sign_samples)
                existing_samples_count[sign_name] = existing_samples_count.get(sign_name, 0) + len(current_sign_samples)
                state = "REVIEW"
            else:
                progress = elapsed / record_duration
                bar_w = int(w * progress)
                cv2.rectangle(annotated_frame, (0, h - 60), (bar_w, h), (16, 185, 129), -1)

                cv2.putText(annotated_frame, f"RECORDING [{sign_name}]: {remaining:.1f}s left | Captured: {len(current_sign_samples)} frames",
                            (25, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(annotated_frame, (w - 40, 45), 14, (0, 0, 255), -1)

        elif state == "REVIEW":
            cv2.rectangle(annotated_frame, (w // 2 - 320, h // 2 - 100), (w // 2 + 320, h // 2 + 100), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (w // 2 - 320, h // 2 - 100), (w // 2 + 320, h // 2 + 100), (16, 185, 129), 3)

            cv2.putText(annotated_frame, f"Completed [{sign_name}]!", (w // 2 - 170, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (16, 185, 129), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Tagged & Saved {len(current_sign_samples)} frames as '{sign_name}'.", (w // 2 - 250, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (241, 245, 249), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Press [SPACE] or [n] for Next Sign | [r] to Re-record", (w // 2 - 280, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA)

        elif state == "PAUSED":
            cv2.rectangle(annotated_frame, (w // 2 - 250, h // 2 - 60), (w // 2 + 250, h // 2 + 60), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (w // 2 - 250, h // 2 - 60), (w // 2 + 250, h // 2 + 60), (245, 158, 11), 3)
            cv2.putText(annotated_frame, "PAUSED - Take your time!", (w // 2 - 180, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 158, 11), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Press [p] or [SPACE] to Resume", (w // 2 - 160, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("ISL Self-Paced Data Collector", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nExiting data collector...")
            break
        elif key == ord('p'):
            if state == "PAUSED":
                state = "PRACTICE"
            elif state == "PRACTICE":
                state = "PAUSED"
        elif key in (ord(' '), ord('s')):
            if state == "PRACTICE":
                state = "COUNTDOWN"
                countdown_start = time.time()
            elif state == "REVIEW":
                if current_idx + 1 < total_signs:
                    current_idx += 1
                    state = "PRACTICE"
                else:
                    print("All signs in pack completed!")
                    break
            elif state == "PAUSED":
                state = "PRACTICE"
        elif key == ord('n'):
            if current_idx + 1 < total_signs:
                current_idx += 1
                state = "PRACTICE"
        elif key == ord('b'):
            if current_idx > 0:
                current_idx -= 1
                state = "PRACTICE"
        elif key == ord('r'):
            state = "PRACTICE"

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    if all_data:
        feature_cols = [f"f_{i}" for i in range(126)]
        cols = feature_cols + ["label"]
        new_df = pd.DataFrame(all_data, columns=cols)

        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            print(f"\nSaved! Added {len(new_df)} new frames. Total dataset size: {len(combined_df)} samples in {output_csv}")
        else:
            new_df.to_csv(output_csv, index=False)
            print(f"\nSaved! Total dataset size: {len(new_df)} samples in {output_csv}")

        print("\nTo update and evaluate your model with this new data, run:")
        print("  python train_model.py\n")
    else:
        print("\nNo new samples recorded in this session.")

if __name__ == "__main__":
    run_data_collection(record_duration=5.0)
