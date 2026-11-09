import os
from rapidfuzz import process, fuzz

CORE_ISL_VOCAB = [
    "HELLO", "NAMASTE", "WELCOME", "PLEASE", "THANK YOU", "SORRY", "GOOD", "BAD",
    "YES", "NO", "I LOVE YOU", "LOVE", "MORNING", "NIGHT", "AFTERNOON", "EVENING", "HELP", "FRIEND",
    "FAMILY", "FATHER", "MOTHER", "BROTHER", "SISTER", "NAME", "WHAT", "WHERE",
    "WHEN", "WHY", "HOW", "WHO", "TIME", "TODAY", "TOMORROW", "YESTERDAY",
    "WATER", "FOOD", "HUNGRY", "THIRSTY", "TIRED", "HAPPY", "SAD", "ANGRY",
    "SCHOOL", "COLLEGE", "TEACHER", "STUDENT", "BOOK", "PEN", "COMPUTER",
    "PROJECT", "LANGUAGE", "SIGN", "INDIA", "LEARN", "WORK", "OFFICE",
    "HOSPITAL", "DOCTOR", "MEDICINE", "EMERGENCY", "POLICE", "HOME", "HOUSE",
    "CAR", "BUS", "TRAIN", "ROAD", "MONEY", "MARKET", "SHOP", "BUY",
    "AGAIN", "AGREE", "ANSWER", "ATTENDANCE", "BREAK", "CAREFUL", "CHANGE",
    "CHAT", "CONGRATULATIONS", "EMAIL", "FILE", "GOOD MORNING", "HAPPY BIRTHDAY",
    "I NEED HELP", "JOIN", "KEEPSMILE", "MEET", "MISTAKE", "OPEN", "OPINION",
    "PASS", "PRACTICE", "PRESSURE", "PROBLEM", "QUESTIONS", "REMEMBER", "SEAT",
    "SHIFT", "SICK", "STOP", "SUN", "TEAM", "THIS", "TOGETHER", "UNDERSTAND",
    "WAIT", "WRITE", "ASSOCIATION", "SMILE", "SMILING"
]

class WordSuggester:
    """
    Intelligent English & ISL Word Suggester:
    1. Loads 82,000+ English frequency dictionary with real-world usage counts.
    2. Exact prefix matches prioritized by natural word frequency.
    3. Typo fuzzy correction (>= 70 similarity) applied only if no prefix matches exist.
    """
    def __init__(self, freq_file="data/frequency_dictionary_en_82k.txt", fallback_file="data/english_words.txt"):
        self.word_freq = {}
        
        # Priority for Core ISL Vocabulary
        for w in CORE_ISL_VOCAB:
            self.word_freq[w] = 10**12

        # Load 82k Frequency Dictionary
        loaded = False
        if os.path.exists(freq_file):
            try:
                with open(freq_file, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            w = parts[0].upper()
                            if w not in self.word_freq and len(w) >= 2:
                                self.word_freq[w] = int(parts[1])
                loaded = True
            except Exception as e:
                print(f"Warning loading {freq_file}: {e}")

        # Fallback to secondary dictionary if needed
        if not loaded and os.path.exists(fallback_file):
            try:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().upper()
                        if w not in self.word_freq and len(w) >= 2:
                            self.word_freq[w] = 1000
            except Exception as e:
                print(f"Warning loading fallback {fallback_file}: {e}")

        # Precompute top words for fast typo correction
        self.top_words = sorted(self.word_freq.keys(), key=lambda w: self.word_freq[w], reverse=True)[:25000]

    def suggest(self, query, top_k=5):
        if not query:
            return []
        
        q = query.strip().upper()
        if not q:
            return []
        
        # 1. Exact Prefix matches first, ranked by (exact match first, then frequency descending)
        prefix_matches = [w for w in self.word_freq if w.startswith(q)]
        if prefix_matches:
            prefix_matches.sort(key=lambda w: (0 if w == q else 1, -self.word_freq[w]))
            return prefix_matches[:top_k]
        
        # 2. Strict typo correction: only if NO prefix match exists
        pool = [w for w in self.top_words if abs(len(w) - len(q)) <= 2]
        if pool:
            fuzzy = process.extract(q, pool, scorer=fuzz.ratio, limit=top_k * 3)
            candidates = [(w, score) for w, score, _ in fuzzy if score >= 70]
            candidates.sort(key=lambda item: (item[1], self.word_freq.get(item[0], 0)), reverse=True)
            return [w for w, _ in candidates][:top_k]
            
        return []

    def add_word(self, word, freq=10**10):
        w = word.strip().upper()
        if w:
            self.word_freq[w] = freq
            if w not in self.top_words:
                self.top_words.append(w)
