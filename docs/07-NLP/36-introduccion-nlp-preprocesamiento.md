# Introducción a NLP y Preprocesamiento de Texto

## 1. ¿Qué es NLP?

**Natural Language Processing (NLP)** es el campo de la IA que permite a las máquinas entender, interpretar y generar lenguaje humano.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE NLP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TEXTO CRUDO                                                                │
│       │                                                                      │
│       ▼                                                                      │
│   ┌─────────────────┐                                                        │
│   │ PREPROCESAMIENTO│  ← Limpieza, normalización, tokenización              │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ REPRESENTACIÓN  │  ← BoW, TF-IDF, Embeddings                            │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │    MODELADO     │  ← Clasificación, NER, Sentiment, etc.                │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│       RESULTADO                                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tareas Principales de NLP

```
┌────────────────────────┬───────────────────────────────────────────────────┐
│        TAREA           │                    DESCRIPCIÓN                     │
├────────────────────────┼───────────────────────────────────────────────────┤
│ Text Classification    │ Categorizar texto (spam/no spam, topic)           │
│ Sentiment Analysis     │ Detectar emoción/opinión (positivo/negativo)      │
│ Named Entity Recog.    │ Identificar entidades (personas, lugares, fechas) │
│ POS Tagging            │ Etiquetar partes del discurso (verbo, sustantivo) │
│ Machine Translation    │ Traducir entre idiomas                            │
│ Question Answering     │ Responder preguntas sobre un texto                │
│ Text Summarization     │ Resumir documentos largos                         │
│ Text Generation        │ Generar texto nuevo (LLMs)                        │
│ Topic Modeling         │ Descubrir temas en colecciones de documentos      │
└────────────────────────┴───────────────────────────────────────────────────┘
```

---

## 2. Preprocesamiento de Texto

El preprocesamiento es **crítico** - texto mal procesado = modelo malo.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PREPROCESAMIENTO                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "Check out THIS link: http://evil.com!!! 😈 FREE $$$"                      │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────┐                                                        │
│  │ 1. Lowercase     │ → "check out this link: http://evil.com!!! 😈 free $$$"│
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. Remove URLs   │ → "check out this link:  😈 free $$$"                 │
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 3. Remove Punct. │ → "check out this link  😈 free "                     │
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. Remove Emojis │ → "check out this link  free "                        │
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 5. Tokenize      │ → ["check", "out", "this", "link", "free"]            │
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 6. Remove Stops  │ → ["check", "link", "free"]                           │
│  └────────┬─────────┘                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 7. Lemmatize     │ → ["check", "link", "free"]                           │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tokenización

**Tokenización**: Dividir texto en unidades más pequeñas (tokens).

### 3.1 Tipos de Tokenización

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NIVELES DE TOKENIZACIÓN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Texto: "I can't believe it's working!"                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ WORD TOKENIZATION                                                    │    │
│  │ → ["I", "can't", "believe", "it's", "working", "!"]                 │    │
│  │ Problema: "can't" = "can" + "not"?                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ SENTENCE TOKENIZATION                                                │    │
│  │ Texto: "Hello. How are you? I'm fine."                              │    │
│  │ → ["Hello.", "How are you?", "I'm fine."]                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ SUBWORD TOKENIZATION (BPE, WordPiece)                               │    │
│  │ → ["I", "can", "'", "t", "believe", "it", "'", "s", "work", "##ing"]│    │
│  │ Usado por BERT, GPT - maneja palabras desconocidas                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ CHARACTER TOKENIZATION                                               │    │
│  │ → ["I", " ", "c", "a", "n", "'", "t", ...]                          │    │
│  │ Vocabulario pequeño pero secuencias muy largas                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementación de Tokenización

```python
import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Descargar recursos de NLTK (solo primera vez)
# nltk.download('punkt')

class TextTokenizer:
    """
    Tokenizador de texto con múltiples estrategias.
    """

    def __init__(self):
        pass

    def word_tokenize_simple(self, text: str) -> List[str]:
        """
        Tokenización simple por espacios y puntuación.
        """
        # Separar por espacios y puntuación
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def word_tokenize_nltk(self, text: str) -> List[str]:
        """
        Tokenización con NLTK (más sofisticada).
        Maneja contracciones, puntuación, etc.
        """
        return word_tokenize(text.lower())

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Divide texto en oraciones.
        """
        return sent_tokenize(text)

    def tokenize_for_security(self, text: str) -> List[str]:
        """
        Tokenización específica para análisis de seguridad.
        Preserva IPs, URLs, hashes, etc.
        """
        # Patrones especiales de seguridad
        patterns = {
            'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'hash_md5': r'\b[a-fA-F0-9]{32}\b',
            'hash_sha256': r'\b[a-fA-F0-9]{64}\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
            'registry': r'HKEY_[A-Z_]+\\[^\s]+',
            'path': r'[A-Za-z]:\\[^\s]+',
        }

        # Extraer tokens especiales primero
        special_tokens = []
        text_clean = text

        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                special_tokens.append((pattern_type, match))
                text_clean = text_clean.replace(match, f' __{pattern_type.upper()}__ ')

        # Tokenizar el resto
        regular_tokens = self.word_tokenize_simple(text_clean)

        return regular_tokens, special_tokens


# Ejemplo de uso
tokenizer = TextTokenizer()

# Texto de ejemplo (log de seguridad)
log_text = """
Failed login attempt from 192.168.1.100 for user admin.
Suspicious file detected: C:\\Windows\\Temp\\malware.exe
Hash: 5d41402abc4b2a76b9719d911017c592
Connection to http://evil.com/payload blocked.
"""

print("=== Tokenización Simple ===")
tokens = tokenizer.word_tokenize_simple(log_text)
print(tokens[:15])

print("\n=== Tokenización NLTK ===")
tokens = tokenizer.word_tokenize_nltk(log_text)
print(tokens[:15])

print("\n=== Tokenización Seguridad ===")
regular, special = tokenizer.tokenize_for_security(log_text)
print(f"Tokens regulares: {regular[:10]}")
print(f"Tokens especiales: {special}")
```

---

## 4. Normalización de Texto

### 4.1 Lowercase

```python
def normalize_case(text: str, preserve_acronyms: bool = False) -> str:
    """
    Convierte a minúsculas, opcionalmente preservando acrónimos.
    """
    if preserve_acronyms:
        # Preservar palabras completamente en mayúsculas (acrónimos)
        words = text.split()
        normalized = []
        for word in words:
            if word.isupper() and len(word) > 1:
                normalized.append(word)  # Mantener acrónimo
            else:
                normalized.append(word.lower())
        return ' '.join(normalized)
    return text.lower()


# Ejemplo
text = "The FBI detected a DDoS attack from APT29"
print(normalize_case(text))
# "the fbi detected a ddos attack from apt29"

print(normalize_case(text, preserve_acronyms=True))
# "the FBI detected a DDoS attack from APT29"
```

### 4.2 Eliminación de Ruido

```python
import re
import unicodedata
from typing import Optional

class TextCleaner:
    """
    Limpieza de texto para NLP.
    """

    @staticmethod
    def remove_urls(text: str, replacement: str = '') -> str:
        """Elimina URLs."""
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, replacement, text)

    @staticmethod
    def remove_emails(text: str, replacement: str = '') -> str:
        """Elimina emails."""
        pattern = r'\b[\w.-]+@[\w.-]+\.\w+\b'
        return re.sub(pattern, replacement, text)

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Elimina tags HTML."""
        pattern = r'<[^>]+>'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_punctuation(text: str, keep: str = '') -> str:
        """
        Elimina puntuación excepto caracteres especificados.
        """
        import string
        punct_to_remove = ''.join(c for c in string.punctuation if c not in keep)
        translator = str.maketrans('', '', punct_to_remove)
        return text.translate(translator)

    @staticmethod
    def remove_numbers(text: str) -> str:
        """Elimina números."""
        return re.sub(r'\d+', '', text)

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Normaliza espacios en blanco."""
        return ' '.join(text.split())

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Elimina emojis."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normaliza caracteres Unicode (acentos, etc.).
        NFD: Descompone caracteres
        Luego elimina marcas diacríticas
        """
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

    def clean_text(self, text: str,
                   lowercase: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_html: bool = True,
                   remove_punct: bool = True,
                   remove_numbers: bool = False,
                   remove_emojis: bool = True,
                   normalize_unicode: bool = False) -> str:
        """
        Pipeline completo de limpieza.
        """
        if remove_html:
            text = self.remove_html_tags(text)
        if remove_urls:
            text = self.remove_urls(text)
        if remove_emails:
            text = self.remove_emails(text)
        if remove_emojis:
            text = self.remove_emojis(text)
        if lowercase:
            text = text.lower()
        if remove_punct:
            text = self.remove_punctuation(text)
        if remove_numbers:
            text = self.remove_numbers(text)
        if normalize_unicode:
            text = self.normalize_unicode(text)

        text = self.remove_extra_whitespace(text)
        return text


# Ejemplo
cleaner = TextCleaner()

messy_text = """
<html>Check out http://spam.com for FREE stuff!!! 😀😀😀
Contact: scammer@evil.org
BUY NOW $$$$ 100% GUARANTEED!!!
</html>
"""

clean = cleaner.clean_text(messy_text)
print(f"Original:\n{messy_text}")
print(f"Limpio:\n{clean}")
# "check out for free stuff contact buy now guaranteed"
```

---

## 5. Stop Words

**Stop words**: Palabras muy comunes que aportan poco significado semántico.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STOP WORDS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INGLÉS: the, a, an, is, are, was, were, be, been, being, have, has, had,   │
│          do, does, did, will, would, could, should, may, might, must,       │
│          this, that, these, those, i, you, he, she, it, we, they, and,      │
│          or, but, if, then, else, when, where, why, how, what, which...     │
│                                                                              │
│  ESPAÑOL: el, la, los, las, un, una, de, del, al, y, o, pero, si, no,       │
│           que, en, por, para, con, sin, sobre, entre, como, más, menos...   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  CUÁNDO ELIMINAR:                                                            │
│    ✓ Clasificación de texto (spam, sentiment)                               │
│    ✓ Topic modeling                                                          │
│    ✓ Keyword extraction                                                      │
│                                                                              │
│  CUÁNDO MANTENER:                                                            │
│    ✗ Traducción automática                                                   │
│    ✗ Question answering                                                      │
│    ✗ Modelos de lenguaje (BERT, GPT)                                        │
│    ✗ Cuando el contexto importa ("to be or not to be")                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación

```python
from nltk.corpus import stopwords
from typing import List, Set

# nltk.download('stopwords')

class StopWordRemover:
    """
    Eliminación de stop words con opciones personalizables.
    """

    def __init__(self, language: str = 'english',
                 custom_stopwords: Set[str] = None,
                 keep_words: Set[str] = None):
        """
        Args:
            language: Idioma para stop words de NLTK
            custom_stopwords: Stop words adicionales
            keep_words: Palabras a mantener aunque sean stop words
        """
        self.stopwords = set(stopwords.words(language))

        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        if keep_words:
            self.stopwords -= keep_words

    def remove(self, tokens: List[str]) -> List[str]:
        """Elimina stop words de una lista de tokens."""
        return [t for t in tokens if t.lower() not in self.stopwords]

    def get_stopwords(self) -> Set[str]:
        """Retorna el conjunto de stop words."""
        return self.stopwords


# Ejemplo: Stop words personalizadas para seguridad
security_stopwords = {
    'http', 'https', 'www', 'com', 'org', 'net',  # Partes de URLs
    'log', 'info', 'debug', 'error', 'warning',   # Niveles de log
}

remover = StopWordRemover(
    language='english',
    custom_stopwords=security_stopwords,
    keep_words={'not', 'no', 'failed'}  # Mantener negaciones
)

tokens = ['the', 'user', 'failed', 'to', 'login', 'from', 'http', 'source']
filtered = remover.remove(tokens)
print(f"Original: {tokens}")
print(f"Filtrado: {filtered}")
# ['user', 'failed', 'login', 'source']
```

---

## 6. Stemming

**Stemming**: Reducir palabras a su raíz (stem) cortando sufijos.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STEMMING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROCESO: Cortar sufijos mediante reglas heurísticas                        │
│                                                                              │
│  Ejemplos:                                                                   │
│    running   → runn                                                          │
│    runs      → run                                                           │
│    runner    → runner → runn  (depende del stemmer)                         │
│    connection → connect                                                      │
│    connected  → connect                                                      │
│    connecting → connect                                                      │
│                                                                              │
│  STEMMERS POPULARES:                                                         │
│  ┌────────────────┬────────────────────────────────────────────────────┐    │
│  │ Porter Stemmer │ Más común, 5 fases de reglas                       │    │
│  │ Snowball       │ Mejorado, soporta múltiples idiomas               │    │
│  │ Lancaster      │ Más agresivo, stems más cortos                    │    │
│  └────────────────┴────────────────────────────────────────────────────┘    │
│                                                                              │
│  PROBLEMAS:                                                                  │
│    • Over-stemming: "university" → "univers" (pierde significado)           │
│    • Under-stemming: "data" y "datum" → stems diferentes                    │
│    • No produce palabras reales: "running" → "runn"                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from typing import List

class Stemmer:
    """
    Wrapper para diferentes stemmers.
    """

    def __init__(self, algorithm: str = 'porter', language: str = 'english'):
        """
        Args:
            algorithm: 'porter', 'snowball', 'lancaster'
            language: Idioma (solo para snowball)
        """
        if algorithm == 'porter':
            self.stemmer = PorterStemmer()
        elif algorithm == 'snowball':
            self.stemmer = SnowballStemmer(language)
        elif algorithm == 'lancaster':
            self.stemmer = LancasterStemmer()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm = algorithm

    def stem(self, word: str) -> str:
        """Aplica stemming a una palabra."""
        return self.stemmer.stem(word)

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Aplica stemming a una lista de tokens."""
        return [self.stem(t) for t in tokens]


# Comparar stemmers
words = ['running', 'runs', 'runner', 'ran',
         'connection', 'connected', 'connecting', 'connections',
         'malware', 'malicious', 'infected', 'infection']

print("Comparación de Stemmers:")
print("-" * 60)
print(f"{'Palabra':<15} {'Porter':<12} {'Snowball':<12} {'Lancaster':<12}")
print("-" * 60)

porter = Stemmer('porter')
snowball = Stemmer('snowball')
lancaster = Stemmer('lancaster')

for word in words:
    p = porter.stem(word)
    s = snowball.stem(word)
    l = lancaster.stem(word)
    print(f"{word:<15} {p:<12} {s:<12} {l:<12}")
```

Output:
```
Palabra         Porter       Snowball     Lancaster
------------------------------------------------------------
running         run          run          run
runs            run          run          run
runner          runner       runner       run
ran             ran          ran          ran
connection      connect      connect      connect
connected       connect      connect      connect
connecting      connect      connect      connect
connections     connect      connect      connect
malware         malwar       malwar       malw
malicious       malici       malici       mal
infected        infect       infect       infect
infection       infect       infect       infect
```

---

## 7. Lemmatization

**Lemmatization**: Reducir palabras a su forma base (lemma) usando análisis morfológico.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEMMING vs LEMMATIZATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┬─────────────────────────────┐              │
│  │         STEMMING            │       LEMMATIZATION          │              │
│  ├─────────────────────────────┼─────────────────────────────┤              │
│  │ Reglas heurísticas          │ Diccionario + Morfología    │              │
│  │ Rápido                      │ Más lento                   │              │
│  │ No requiere contexto        │ Puede usar POS tags         │              │
│  │ Produce stems (no palabras) │ Produce palabras válidas    │              │
│  └─────────────────────────────┴─────────────────────────────┘              │
│                                                                              │
│  EJEMPLOS:                                                                   │
│                                                                              │
│  Palabra      Stemming      Lemmatization                                   │
│  ─────────────────────────────────────────                                  │
│  running      runn          run (verbo)                                     │
│  better       better        good (adjetivo)                                 │
│  ran          ran           run (verbo)                                     │
│  mice         mice          mouse (sustantivo)                              │
│  are          ar            be (verbo)                                      │
│  studies      studi         study (verbo/sustantivo)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from typing import List, Tuple

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

class Lemmatizer:
    """
    Lemmatizador con soporte para POS tags.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convierte POS tag de Penn Treebank a WordNet.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default

    def lemmatize(self, word: str, pos: str = None) -> str:
        """
        Lemmatiza una palabra.

        Args:
            word: Palabra a lemmatizar
            pos: POS tag opcional ('n', 'v', 'a', 'r')
        """
        if pos:
            return self.lemmatizer.lemmatize(word, pos)
        return self.lemmatizer.lemmatize(word)

    def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
        """
        Lemmatiza tokens usando POS tagging automático.
        Más preciso que lemmatizar sin contexto.
        """
        # Obtener POS tags
        pos_tags = pos_tag(tokens)

        lemmas = []
        for word, tag in pos_tags:
            wordnet_pos = self._get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(word.lower(), wordnet_pos)
            lemmas.append(lemma)

        return lemmas


# Ejemplo
lemmatizer = Lemmatizer()

# Sin POS (asume sustantivo)
words = ['running', 'better', 'mice', 'are', 'studies', 'went']
print("Sin POS tag (asume sustantivo):")
for word in words:
    lemma = lemmatizer.lemmatize(word)
    print(f"  {word} → {lemma}")

# Con POS automático
sentence = "The hackers are running malicious scripts that infected multiple systems"
tokens = sentence.lower().split()

print("\nCon POS tag automático:")
lemmas = lemmatizer.lemmatize_with_pos(tokens)
for token, lemma in zip(tokens, lemmas):
    if token != lemma:
        print(f"  {token} → {lemma}")
```

---

## 8. N-gramas

**N-gramas**: Secuencias de N tokens consecutivos.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            N-GRAMAS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Texto: "the quick brown fox"                                               │
│                                                                              │
│  Unigramas (n=1): ["the", "quick", "brown", "fox"]                          │
│                                                                              │
│  Bigramas (n=2):  ["the quick", "quick brown", "brown fox"]                 │
│                                                                              │
│  Trigramas (n=3): ["the quick brown", "quick brown fox"]                    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ¿POR QUÉ USAR N-GRAMAS?                                                    │
│                                                                              │
│  "New York" como unigramas: "new" + "york" (pierde significado)             │
│  "New York" como bigrama: "new york" (captura la entidad)                   │
│                                                                              │
│  "credit card fraud" necesita trigramas para capturar el concepto           │
│                                                                              │
│  En seguridad:                                                               │
│    "buffer overflow" → bigrama importante                                   │
│    "remote code execution" → trigrama importante                            │
│    "command and control" → trigrama (C2)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación

```python
from nltk import ngrams
from collections import Counter
from typing import List, Tuple

def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extrae n-gramas de una lista de tokens."""
    return list(ngrams(tokens, n))

def extract_ngrams_range(tokens: List[str],
                         min_n: int = 1,
                         max_n: int = 3) -> List[str]:
    """
    Extrae n-gramas en un rango (ej: 1-3).
    Retorna strings unidos.
    """
    all_ngrams = []
    for n in range(min_n, max_n + 1):
        for gram in ngrams(tokens, n):
            all_ngrams.append(' '.join(gram))
    return all_ngrams

def get_top_ngrams(texts: List[str], n: int, top_k: int = 10) -> List[Tuple[str, int]]:
    """
    Encuentra los n-gramas más frecuentes en un corpus.
    """
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        grams = [' '.join(g) for g in ngrams(tokens, n)]
        all_ngrams.extend(grams)

    return Counter(all_ngrams).most_common(top_k)


# Ejemplo con texto de seguridad
security_texts = [
    "buffer overflow vulnerability detected in web application",
    "sql injection attack exploiting buffer overflow",
    "remote code execution via buffer overflow exploit",
    "cross site scripting xss vulnerability found",
    "cross site scripting attack blocked",
    "command and control server detected",
    "malware connecting to command and control",
]

print("Top bigramas en textos de seguridad:")
for gram, count in get_top_ngrams(security_texts, 2, 5):
    print(f"  '{gram}': {count}")

print("\nTop trigramas:")
for gram, count in get_top_ngrams(security_texts, 3, 5):
    print(f"  '{gram}': {count}")
```

---

## 9. Pipeline Completo de Preprocesamiento

```python
import re
from typing import List, Optional
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

@dataclass
class PreprocessingConfig:
    """Configuración del preprocesamiento."""
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_token_length: int = 2
    language: str = 'english'


class NLPPreprocessor:
    """
    Pipeline completo de preprocesamiento de texto.
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words(self.config.language))

    def preprocess(self, text: str) -> List[str]:
        """
        Aplica el pipeline completo de preprocesamiento.

        Returns:
            Lista de tokens procesados
        """
        # 1. Limpieza inicial
        if self.config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        if self.config.remove_emails:
            text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '', text)

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # 2. Tokenización
        tokens = word_tokenize(text)

        # 3. Filtrado por longitud
        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        # 4. Stop words
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # 5. Lemmatización
        if self.config.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Procesa múltiples textos."""
        return [self.preprocess(text) for text in texts]

    def preprocess_to_string(self, text: str) -> str:
        """Retorna texto preprocesado como string."""
        tokens = self.preprocess(text)
        return ' '.join(tokens)


# Ejemplo de uso
config = PreprocessingConfig(
    lowercase=True,
    remove_urls=True,
    remove_stopwords=True,
    lemmatize=True,
    min_token_length=2
)

preprocessor = NLPPreprocessor(config)

# Texto de ejemplo (email sospechoso)
email = """
Subject: URGENT: Your account has been compromised!!!

Dear valued customer,

We detected suspicious activities in your account.
Click here immediately: http://phishing-site.com/login
Or contact us at support@fake-bank.com

Your account will be SUSPENDED within 24 hours!!!

Thanks,
Security Team
"""

tokens = preprocessor.preprocess(email)
print("Tokens procesados:")
print(tokens)

# Para TF-IDF/modelo
clean_text = preprocessor.preprocess_to_string(email)
print(f"\nTexto limpio:\n{clean_text}")
```

---

## 10. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESAMIENTO NLP - RESUMEN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PIPELINE TÍPICO:                                                            │
│    1. Limpieza (URLs, HTML, caracteres especiales)                          │
│    2. Normalización (lowercase, unicode)                                    │
│    3. Tokenización (dividir en palabras/subpalabras)                        │
│    4. Stop words (eliminar palabras comunes)                                │
│    5. Stemming/Lemmatization (reducir a forma base)                         │
│    6. N-gramas (capturar frases)                                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEMMING vs LEMMATIZATION:                                                  │
│    • Stemming: Rápido, reglas, no produce palabras reales                   │
│    • Lemmatization: Lento, diccionario, produce palabras válidas            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DECISIONES IMPORTANTES:                                                     │
│                                                                              │
│    ¿Eliminar números?                                                        │
│      → Sí para sentiment, No para detección de IPs/puertos                  │
│                                                                              │
│    ¿Eliminar stop words?                                                     │
│      → Sí para clasificación, No para LLMs/traducción                       │
│                                                                              │
│    ¿Stemming o Lemmatization?                                                │
│      → Stemming para velocidad, Lemmatization para precisión                │
│                                                                              │
│    ¿Qué n-gramas?                                                            │
│      → (1,1) para vocabulario simple                                        │
│      → (1,2) para capturar bigramas importantes                             │
│      → (1,3) si hay trigramas relevantes (más features)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
