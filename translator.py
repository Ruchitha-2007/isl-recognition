import urllib.request
import urllib.parse
import json

SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "voice_code": "en-IN"},
    "hi": {"name": "Hindi", "native": "हिंदी", "voice_code": "hi-IN"},
    "te": {"name": "Telugu", "native": "తెలుగు", "voice_code": "te-IN"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "voice_code": "ta-IN"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "voice_code": "kn-IN"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "voice_code": "ml-IN"},
    "mr": {"name": "Marathi", "native": "मराठी", "voice_code": "mr-IN"},
    "bn": {"name": "Bengali", "native": "বাংলা", "voice_code": "bn-IN"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "voice_code": "gu-IN"},
    "es": {"name": "Spanish", "native": "Español", "voice_code": "es-ES"},
    "fr": {"name": "French", "native": "Français", "voice_code": "fr-FR"},
    "de": {"name": "German", "native": "Deutsch", "voice_code": "de-DE"},
}

class LanguageTranslator:
    def __init__(self):
        self.cache = {}

    def translate(self, text: str, target_lang: str = "en") -> str:
        text = text.strip()
        if not text:
            return ""
        
        target_lang = target_lang.lower().strip()
        if target_lang in ("en", "auto"):
            return text

        cache_key = (text, target_lang)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                translated = "".join([part[0] for part in data[0] if part and part[0]])
                if translated:
                    self.cache[cache_key] = translated
                    return translated
        except Exception as e:
            print(f"Translation error ({target_lang}): {e}")

        return text

    def get_tts_audio(self, text: str, lang: str = "en"):
        text = text.strip()
        if not text:
            return None
        lang = lang.lower().strip()
        try:
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                return resp.read()
        except Exception as e:
            print(f"TTS audio error ({lang}): {e}")
            return None
