import random
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Класс для аугментации текстов Аристотеля"""
    
    def __init__(self):
        self.synonyms = {
            "мудрость": ["разум", "знание", "понимание"],
            "добродетель": ["благо", "достоинство", "нравственность"],
            "истина": ["правда", "реальность", "действительность"],
            "справедливость": ["правосудие", "честность", "беспристрастность"],
            "счастье": ["благополучие", "блаженство", "удовлетворение"],
            "душа": ["дух", "сущность", "природа"],
            "разум": ["ум", "интеллект", "рассудок"],
            "знание": ["познание", "ведение", "осведомленность"]
        }
        
        self.philosophical_connectors = [
            "следовательно", "таким образом", "поэтому", "итак",
            "однако", "тем не менее", "впрочем", "между тем",
            "более того", "кроме того", "помимо этого",
            "с другой стороны", "напротив", "в противоположность"
        ]
    
    def augment_text(self, text: str) -> str:
        """Основной метод аугментации текста"""
        if not text or len(text.strip()) < 10:
            return text
            
        # Применяем различные техники аугментации
        augmented = text
        
        # 30% вероятность каждой техники
        if random.random() < 0.3:
            augmented = self._synonym_replacement(augmented)
            
        if random.random() < 0.3:
            augmented = self._add_philosophical_connectors(augmented)
            
        if random.random() < 0.2:
            augmented = self._sentence_reordering(augmented)
            
        return augmented
    
    def _synonym_replacement(self, text: str) -> str:
        """Замена слов синонимами"""
        words = text.split()
        
        for i, word in enumerate(words):
            # ��чищаем слово от пунктуации для поиска
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.synonyms and random.random() < 0.3:
                synonym = random.choice(self.synonyms[clean_word])
                # Сохраняем регистр и пунктуацию
                if word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Заменяем только основу слова, сохраняя пунктуацию
                words[i] = re.sub(re.escape(clean_word), synonym, word, flags=re.IGNORECASE)
        
        return ' '.join(words)
    
    def _add_philosophical_connectors(self, text: str) -> str:
        """Добавление философских связок"""
        sentences = text.split('.')
        
        if len(sentences) < 2:
            return text
            
        # Добавляем связку между случайными предложениями
        insert_pos = random.randint(1, len(sentences) - 1)
        connector = random.choice(self.philosophical_connectors)
        
        if sentences[insert_pos].strip():
            sentences[insert_pos] = f" {connector}, {sentences[insert_pos].strip()}"
        
        return '.'.join(sentences)
    
    def _sentence_reordering(self, text: str) -> str:
        """Перестановка предложений (осторожно, чтобы не нарушить логику)"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 3:
            return text
            
        # Меняем местами только соседние предложения
        if random.random() < 0.5 and len(sentences) >= 2:
            idx = random.randint(0, len(sentences) - 2)
            sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
        
        return '. '.join(sentences) + '.'
    
    def batch_augment(self, texts: List[str], augment_factor: int = 2) -> List[str]:
        """Пакетная аугментация текстов"""
        augmented_texts = []
        
        for text in texts:
            augmented_texts.append(text)  # Оригинальный текст
            
            # Создаем дополнительные варианты
            for _ in range(augment_factor - 1):
                augmented = self.augment_text(text)
                augmented_texts.append(augmented)
        
        logger.info(f"Augmented {len(texts)} texts to {len(augmented_texts)} texts")
        return augmented_texts