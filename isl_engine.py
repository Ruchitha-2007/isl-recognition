import cv2
import numpy as np

# Robust cross-platform MediaPipe import (guaranteed to work across Linux/Render/Windows)
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    try:
        from mediapipe.solutions import hands as mp_hands
        from mediapipe.solutions import drawing_utils as mp_draw
        from mediapipe.solutions import drawing_styles as mp_drawing_styles
    except (ImportError, AttributeError):
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(p1, p2, p3):
    """Calculates angle (in degrees) at joint p2 formed by lines (p1-p2) and (p3-p2)."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

class ISLLandmarkExtractor:
    """
    Robust Dual-Hand Feature Extractor for Indian Sign Language (ISL).
    Features extracted:
    1. 3D Coordinates normalized to wrist & scaled by palm size.
    2. Rotation-invariant finger joint flexion angles (Thumb, Index, Middle, Ring, Pinky).
    3. Inter-fingertip distance ratios.
    4. Two-hand relative distance metrics (when both hands are present).
    
    Hands are sorted deterministically from Left-to-Right by their X position,
    avoiding MediaPipe left/right label flipping bugs on webcams.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.4):
        self.mp_hands = mp_hands
        self.mp_draw = mp_draw
        self.mp_drawing_styles = mp_drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def _extract_single_hand(self, landmarks_3d):
        """
        Extracts normalized coordinates + joint angles + fingertip distances for one hand (21 points).
        Returns a 1D vector of fixed length 78.
        """
        # 1. Coordinate normalization (Wrist at 0,0,0)
        coords = np.array(landmarks_3d, dtype=np.float32) # shape: (21, 3)
        wrist = coords[0].copy()
        norm_coords = coords - wrist
        
        # Scale normalization by palm size (wrist to middle MCP index 9)
        palm_scale = np.linalg.norm(norm_coords[9])
        if palm_scale < 1e-4:
            palm_scale = 1.0
        norm_coords = norm_coords / palm_scale

        # 2. Joint Angles (Rotation-invariant finger flexion)
        # 5 fingers: Thumb (1-2-3, 2-3-4), Index (5-6-7, 6-7-8), Middle (9-10-11, 10-11-12),
        # Ring (13-14-15, 14-15-16), Pinky (17-18-19, 18-19-20)
        joint_triplets = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),      # Thumb
            (0, 5, 6), (5, 6, 7), (6, 7, 8),      # Index
            (0, 9, 10), (9, 10, 11), (10, 11, 12),# Middle
            (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
            (0, 17, 18), (17, 18, 19), (18, 19, 20) # Pinky
        ]
        angles = []
        for p1_i, p2_i, p3_i in joint_triplets:
            ang = calculate_angle(coords[p1_i], coords[p2_i], coords[p3_i])
            angles.append(ang / 180.0) # normalize to [0, 1]

        # 3. Inter-fingertip distances normalized by palm scale
        # Tips: Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20
        tips = [4, 8, 12, 16, 20]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(norm_coords[tips[i]] - norm_coords[tips[j]])
                distances.append(d)

        # Vector: 63 coordinates + 15 angles + 10 distances = 88 features
        return np.concatenate([norm_coords.flatten(), angles, distances])

    def extract_landmarks(self, frame_bgr):
        """
        Processes frame and returns:
        - feature_vector: 1D numpy array of length 180 (2 hands * 88 + 4 two-hand interaction features)
        - annotated_frame: frame with drawn skeleton
        - hands_detected: int (0, 1, or 2)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame_bgr.copy()
        hands_detected = 0
        
        # Single hand length = 88 features
        hand1_features = np.zeros(88, dtype=np.float32)
        hand2_features = np.zeros(88, dtype=np.float32)
        two_hand_interaction = np.zeros(4, dtype=np.float32) # relative wrist dist, relative tip dist

        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            raw_hands = []
            
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                coords_3d = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
                wrist_x = hand_lms.landmark[0].x
                raw_hands.append((wrist_x, coords_3d))
                
            # Deterministic Sorting: Hand with smaller X (left in image) is Hand 1, other is Hand 2
            raw_hands.sort(key=lambda item: item[0])
            
            if len(raw_hands) == 1:
                hand1_features = self._extract_single_hand(raw_hands[0][1])
            else:
                # 2 Hands present
                hand1_features = self._extract_single_hand(raw_hands[0][1])
                hand2_features = self._extract_single_hand(raw_hands[1][1])
                
                # Compute two-hand interaction features (distance between wrists and index tips)
                h1_wrist = np.array(raw_hands[0][1][0])
                h2_wrist = np.array(raw_hands[1][1][0])
                h1_tip = np.array(raw_hands[0][1][8]) # index tip
                h2_tip = np.array(raw_hands[1][1][8])
                
                two_hand_interaction[0] = np.linalg.norm(h1_wrist - h2_wrist)
                two_hand_interaction[1] = np.linalg.norm(h1_tip - h2_tip)
                two_hand_interaction[2] = abs(h1_wrist[0] - h2_wrist[0])
                two_hand_interaction[3] = abs(h1_wrist[1] - h2_wrist[1])

        feature_vector = np.concatenate([hand1_features, hand2_features, two_hand_interaction])
        return feature_vector, annotated_frame, hands_detected

    def close(self):
        self.hands.close()
