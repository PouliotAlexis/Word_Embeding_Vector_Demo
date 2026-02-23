from deep_translator import GoogleTranslator
from typing import List, Union

class TranslationManager:
    """Manages translation between French and English using deep-translator."""
    def __init__(self):
        # We initialize translators for 'fr' -> 'en' and 'en' -> 'fr'
        self.to_en_translators = {}
        self.from_en_translators = {}

    def _get_to_en_translator(self, source_lang: str):
        if source_lang not in self.to_en_translators:
            self.to_en_translators[source_lang] = GoogleTranslator(source=source_lang, target='en')
        return self.to_en_translators[source_lang]

    def _get_from_en_translator(self, target_lang: str):
        if target_lang not in self.from_en_translators:
            self.from_en_translators[target_lang] = GoogleTranslator(source='en', target=target_lang)
        return self.from_en_translators[target_lang]

    def translate_to_en(self, text: Union[str, List[str]], source_lang: str = "fr") -> Union[str, List[str]]:
        """Translate text or a list of words from a source language to English."""
        if source_lang == "en" or not text:
            return text

        translator = self._get_to_en_translator(source_lang)
        
        try:
            if isinstance(text, list):
                if not text:
                    return []
                # deep-translator handles batch translation natively when passing a list
                translated = translator.translate_batch(text)
                return [t.lower() for t in translated]
            else:
                translated = translator.translate(text)
                return translated.lower() if translated else ""
        except Exception as e:
            print(f"Translation to en error: {e}")
            return text


    def translate_from_en(self, text: Union[str, List[str]], target_lang: str = "fr") -> Union[str, List[str]]:
        """Translate text or a list of words from English to a target language."""
        if target_lang == "en" or not text:
            return text

        translator = self._get_from_en_translator(target_lang)
        
        try:
            if isinstance(text, list):
                if not text:
                    return []
                translated = translator.translate_batch(text)
                return [t.lower() if t else "" for t in translated]
            else:
                translated = translator.translate(text)
                return translated.lower() if translated else ""
        except Exception as e:
            print(f"Translation from en error: {e}")
            return text

# Singleton instance
_manager = None

def get_translation_manager() -> TranslationManager:
    global _manager
    if _manager is None:
        _manager = TranslationManager()
    return _manager
