from rapidfuzz import process, fuzz
import re

# Comprehensive dictionary of common English words and Indian Sign Language vocabulary
DEFAULT_VOCABULARY = [
    # ISL & Conversational Core Words
    "HELLO", "NAMASTE", "WELCOME", "PLEASE", "THANK YOU", "SORRY", "GOOD", "BAD",
    "YES", "NO", "I LOVE YOU", "LOVE", "MORNING", "NIGHT", "AFTERNOON", "EVENING", "HELP", "FRIEND",
    "FAMILY", "FATHER", "MOTHER", "BROTHER", "SISTER", "NAME", "WHAT", "WHERE",
    "WHEN", "WHY", "HOW", "WHO", "TIME", "TODAY", "TOMORROW", "YESTERDAY",
    "WATER", "FOOD", "HUNGRY", "THIRSTY", "TIRED", "HAPPY", "SAD", "ANGRY",
    "SCHOOL", "COLLEGE", "TEACHER", "STUDENT", "BOOK", "PEN", "COMPUTER",
    "PROJECT", "LANGUAGE", "SIGN", "INDIA", "LEARN", "WORK", "OFFICE",
    "HOSPITAL", "DOCTOR", "MEDICINE", "EMERGENCY", "POLICE", "HOME", "HOUSE",
    "CAR", "BUS", "TRAIN", "ROAD", "MONEY", "MARKET", "SHOP", "BUY",
    
    # Common high-frequency English words
    "ABOUT", "AFTER", "AGAIN", "ALL", "ALSO", "AND", "ANY", "ARE", "BACK",
    "BECAUSE", "BEFORE", "BEING", "BEST", "BETTER", "BETWEEN", "BOTH", "CALL",
    "CAME", "CAN", "CHANGE", "CHILD", "COME", "COULD", "DAY", "DEVELOP",
    "DOES", "DONE", "DOWN", "EACH", "EARLY", "EVEN", "EVERY", "FIND", "FIRST",
    "FOR", "FROM", "GIVE", "GREAT", "HAVE", "HEAR", "HERE", "HIGH", "IMPORTANT",
    "INTO", "JUST", "KNOW", "LARGE", "LAST", "LATER", "LIFE", "LIKE", "LITTLE",
    "LOOK", "MAKE", "MANY", "MEET", "MORE", "MOST", "MUCH", "MUST", "NEVER",
    "NEXT", "NIGHT", "NUMBER", "ONLY", "OPEN", "OTHER", "OVER", "PART", "PEOPLE",
    "PLACE", "POINT", "POSSIBLE", "PRESENT", "PROBLEM", "PUBLIC", "QUESTION",
    "READ", "REAL", "RIGHT", "SAME", "SEEM", "SHOW", "SMALL", "SOME", "START",
    "STILL", "SUCH", "SYSTEM", "TAKE", "TELL", "THAN", "THAT", "THEIR", "THERE",
    "THESE", "THING", "THINK", "THIS", "THOSE", "THROUGH", "TOGETHER", "TURN",
    "UNDER", "UNTIL", "UPON", "VERY", "WANT", "WATER", "WELL", "WENT", "WERE",
    "WHICH", "WHILE", "WHITE", "WHOLE", "WITH", "WITHOUT", "WORD", "WORLD",
    "WOULD", "WRITE", "YEAR", "YOUNG"
]

class WordSuggester:
    """
    Intelligent word suggestion engine:
    1. Prefix Autocomplete: Suggests words beginning with the currently spelled characters.
    2. Fuzzy Typo Matching: Uses Levenshtein distance to find nearest matches if sign was misdetected.
    """
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = sorted(list(set(DEFAULT_VOCABULARY)))
        else:
            self.vocab = sorted(list(set(vocab)))

    def suggest(self, query, top_k=5):
        """
        Given a current query string (e.g. 'COMP' or 'HELO'), returns top_k matching words.
        """
        if not query:
            return []
        
        query = query.strip().upper()
        if not query:
            return []
        
        # 1. Exact Prefix matches first
        prefix_matches = [w for w in self.vocab if w.startswith(query)]
        
        # Sort prefix matches by length (shortest first)
        prefix_matches.sort(key=len)
        
        if len(prefix_matches) >= top_k:
            return prefix_matches[:top_k]
        
        # 2. Fuzzy matches for spelling correction using RapidFuzz
        results = process.extract(
            query,
            self.vocab,
            scorer=fuzz.WRatio,
            limit=top_k * 2
        )
        
        suggestions = list(prefix_matches)
        for match_word, score, _ in results:
            if match_word not in suggestions and score >= 60:
                suggestions.append(match_word)
            if len(suggestions) >= top_k:
                break
                
        return suggestions[:top_k]

    def add_word(self, word):
        word = word.strip().upper()
        if word and word not in self.vocab:
            self.vocab.append(word)
            self.vocab.sort()
