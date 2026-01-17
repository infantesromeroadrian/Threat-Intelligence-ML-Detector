# Named Entity Recognition (NER) y POS Tagging

## 1. Tareas de Etiquetado de Secuencias

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  SEQUENCE LABELING TASKS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input:  Una secuencia de tokens                                            │
│  Output: Una etiqueta POR cada token                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ POS TAGGING (Part-of-Speech)                                         │    │
│  │                                                                      │    │
│  │ "The   cat   sat   on   the   mat"                                   │    │
│  │  DET  NOUN  VERB  ADP  DET  NOUN                                    │    │
│  │                                                                      │    │
│  │ Identifica la función gramatical de cada palabra                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ NER (Named Entity Recognition)                                       │    │
│  │                                                                      │    │
│  │ "John   works   at   Google   in   New   York"                       │    │
│  │ B-PER    O      O   B-ORG    O   B-LOC  I-LOC                       │    │
│  │                                                                      │    │
│  │ Identifica entidades nombradas (personas, lugares, organizaciones)  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ CHUNKING                                                              │    │
│  │                                                                      │    │
│  │ "[The cat] [sat] [on the mat]"                                       │    │
│  │    NP      VP       PP                                               │    │
│  │                                                                      │    │
│  │ Agrupa tokens en frases (noun phrase, verb phrase, etc.)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Part-of-Speech (POS) Tagging

### 2.1 Conjunto de Etiquetas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PENN TREEBANK POS TAGS                                    │
├───────────┬───────────────────────────┬─────────────────────────────────────┤
│    Tag    │        Descripción        │            Ejemplos                  │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│    NN     │ Noun, singular            │ cat, dog, computer                  │
│    NNS    │ Noun, plural              │ cats, dogs, computers               │
│    NNP    │ Proper noun, singular     │ John, Google, London                │
│    NNPS   │ Proper noun, plural       │ Americans, Googlers                 │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│    VB     │ Verb, base form           │ run, eat, be                        │
│    VBD    │ Verb, past tense          │ ran, ate, was                       │
│    VBG    │ Verb, gerund              │ running, eating, being              │
│    VBN    │ Verb, past participle     │ run, eaten, been                    │
│    VBP    │ Verb, non-3rd person      │ run, eat, am                        │
│    VBZ    │ Verb, 3rd person sing.    │ runs, eats, is                      │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│    JJ     │ Adjective                 │ big, red, fast                      │
│    JJR    │ Adjective, comparative    │ bigger, redder, faster              │
│    JJS    │ Adjective, superlative    │ biggest, reddest, fastest           │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│    RB     │ Adverb                    │ quickly, very, well                 │
│    RBR    │ Adverb, comparative       │ faster, better                      │
│    RBS    │ Adverb, superlative       │ fastest, best                       │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│    DT     │ Determiner                │ the, a, an, this, that              │
│    IN     │ Preposition/conjunction   │ in, on, at, for, with               │
│    CC     │ Coordinating conjunction  │ and, or, but                        │
│    PRP    │ Personal pronoun          │ I, you, he, she, it                 │
│    PRP$   │ Possessive pronoun        │ my, your, his, her                  │
└───────────┴───────────────────────────┴─────────────────────────────────────┘
```

### 2.2 POS Tagging con NLTK

```python
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import treebank
from typing import List, Tuple

# Descargar recursos necesarios
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

class POSTagger:
    """
    Part-of-Speech Tagger usando NLTK.
    """

    def __init__(self, tagset: str = 'default'):
        """
        Args:
            tagset: 'default' (Penn Treebank) o 'universal'
        """
        self.tagset = tagset

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Etiqueta las palabras de un texto.

        Returns:
            Lista de tuplas (word, tag)
        """
        tokens = word_tokenize(text)

        if self.tagset == 'universal':
            return pos_tag(tokens, tagset='universal')
        return pos_tag(tokens)

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Etiqueta una lista de tokens."""
        if self.tagset == 'universal':
            return pos_tag(tokens, tagset='universal')
        return pos_tag(tokens)

    def get_words_by_tag(self, text: str, target_tag: str) -> List[str]:
        """
        Extrae palabras con una etiqueta específica.
        """
        tagged = self.tag(text)
        return [word for word, tag in tagged if tag.startswith(target_tag)]


# Ejemplo
tagger = POSTagger()

text = "The hacker exploited a critical vulnerability in the system"
tagged = tagger.tag(text)

print("POS Tags (Penn Treebank):")
for word, tag in tagged:
    print(f"  {word:15} → {tag}")

# Con tagset universal (más simple)
tagger_universal = POSTagger(tagset='universal')
tagged_uni = tagger_universal.tag(text)

print("\nPOS Tags (Universal):")
for word, tag in tagged_uni:
    print(f"  {word:15} → {tag}")

# Extraer solo verbos
verbs = tagger.get_words_by_tag(text, 'VB')
print(f"\nVerbos encontrados: {verbs}")
```

### 2.3 POS Tagging con spaCy

```python
import spacy
from typing import List, Dict

class SpacyPOSTagger:
    """
    POS Tagger usando spaCy (más rápido y preciso).
    """

    def __init__(self, model: str = 'en_core_web_sm'):
        """
        Args:
            model: Modelo de spaCy
                   - en_core_web_sm: Pequeño, rápido
                   - en_core_web_md: Medio, con word vectors
                   - en_core_web_lg: Grande, más preciso
        """
        self.nlp = spacy.load(model)

    def tag(self, text: str) -> List[Dict]:
        """
        Etiqueta texto con información detallada.
        """
        doc = self.nlp(text)

        results = []
        for token in doc:
            results.append({
                'text': token.text,
                'pos': token.pos_,      # POS tag universal
                'tag': token.tag_,      # POS tag detallado
                'dep': token.dep_,      # Dependencia sintáctica
                'lemma': token.lemma_,  # Forma base
            })

        return results

    def get_noun_phrases(self, text: str) -> List[str]:
        """Extrae noun phrases."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def get_verbs(self, text: str) -> List[str]:
        """Extrae verbos."""
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ == 'VERB']


# Ejemplo
# Primero: python -m spacy download en_core_web_sm
tagger = SpacyPOSTagger()

text = "The sophisticated malware was rapidly spreading through the network"
results = tagger.tag(text)

print("POS Tags con spaCy:")
print(f"{'Token':<15} {'POS':<8} {'TAG':<8} {'DEP':<12} {'LEMMA':<15}")
print("-" * 60)
for r in results:
    print(f"{r['text']:<15} {r['pos']:<8} {r['tag']:<8} {r['dep']:<12} {r['lemma']:<15}")

print(f"\nNoun Phrases: {tagger.get_noun_phrases(text)}")
print(f"Verbos: {tagger.get_verbs(text)}")
```

---

## 3. Named Entity Recognition (NER)

### 3.1 Esquema de Etiquetado BIO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESQUEMA BIO/IOB                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  B = Beginning (inicio de entidad)                                          │
│  I = Inside (continuación de entidad)                                       │
│  O = Outside (no es entidad)                                                │
│                                                                              │
│  EJEMPLO:                                                                    │
│                                                                              │
│  "John Smith works at Google Inc in New York City"                          │
│                                                                              │
│  John    → B-PER   (Inicio de PERSONA)                                      │
│  Smith   → I-PER   (Continuación de PERSONA)                                │
│  works   → O       (No es entidad)                                          │
│  at      → O       (No es entidad)                                          │
│  Google  → B-ORG   (Inicio de ORGANIZACIÓN)                                 │
│  Inc     → I-ORG   (Continuación de ORGANIZACIÓN)                           │
│  in      → O       (No es entidad)                                          │
│  New     → B-LOC   (Inicio de UBICACIÓN)                                    │
│  York    → I-LOC   (Continuación de UBICACIÓN)                              │
│  City    → I-LOC   (Continuación de UBICACIÓN)                              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VARIANTE BIOES (más precisa):                                              │
│                                                                              │
│  B = Beginning                                                               │
│  I = Inside                                                                  │
│  O = Outside                                                                 │
│  E = End (fin de entidad multi-token)                                       │
│  S = Single (entidad de un solo token)                                      │
│                                                                              │
│  "John Smith works at IBM"                                                  │
│  John  → B-PER                                                              │
│  Smith → E-PER   (fin de entidad)                                          │
│  works → O                                                                  │
│  at    → O                                                                  │
│  IBM   → S-ORG   (entidad de un solo token)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Tipos de Entidades Estándar

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIPOS DE ENTIDADES COMUNES                                │
├───────────┬───────────────────────────┬─────────────────────────────────────┤
│   Tipo    │        Descripción        │            Ejemplos                  │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│   PER     │ Persona                   │ John Smith, Elon Musk               │
│   ORG     │ Organización              │ Google, FBI, United Nations         │
│   LOC     │ Ubicación                 │ New York, Mount Everest             │
│   GPE     │ Entidad geopolítica       │ France, California, Tokyo           │
│   DATE    │ Fecha                     │ January 2024, next Monday           │
│   TIME    │ Hora                      │ 3:00 PM, noon                        │
│   MONEY   │ Valor monetario           │ $100, 50 euros                       │
│   PERCENT │ Porcentaje                │ 50%, twenty percent                  │
│   EVENT   │ Evento                    │ World War II, Olympics              │
│   PRODUCT │ Producto                  │ iPhone, Windows 11                   │
│   WORK    │ Obra (libro, película)    │ Harry Potter, The Matrix            │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│           │   ENTIDADES DE SEGURIDAD  │                                     │
├───────────┼───────────────────────────┼─────────────────────────────────────┤
│   IP      │ Dirección IP              │ 192.168.1.1, 10.0.0.0/8             │
│   URL     │ URL/Dominio               │ evil.com, http://malware.net        │
│   HASH    │ Hash de archivo           │ 5d41402abc4b2a76b9719d911017c592    │
│   CVE     │ Identificador CVE         │ CVE-2021-44228                       │
│   MALWARE │ Nombre de malware         │ WannaCry, Emotet, TrickBot          │
│   APT     │ Grupo APT                 │ APT29, Lazarus Group                │
│   TOOL    │ Herramienta               │ Mimikatz, Cobalt Strike             │
└───────────┴───────────────────────────┴─────────────────────────────────────┘
```

### 3.3 NER con spaCy

```python
import spacy
from spacy import displacy
from typing import List, Dict

class NERExtractor:
    """
    Extractor de Named Entities usando spaCy.
    """

    def __init__(self, model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[Dict]:
        """
        Extrae entidades nombradas del texto.
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })

        return entities

    def extract_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Agrupa entidades por tipo.
        """
        doc = self.nlp(text)

        by_type = {}
        for ent in doc.ents:
            if ent.label_ not in by_type:
                by_type[ent.label_] = []
            by_type[ent.label_].append(ent.text)

        return by_type

    def visualize(self, text: str):
        """
        Visualiza entidades en HTML.
        """
        doc = self.nlp(text)
        return displacy.render(doc, style='ent')


# Ejemplo
ner = NERExtractor()

text = """
Microsoft announced on Tuesday that it will acquire Activision Blizzard
for $68.7 billion. The deal, expected to close in 2023, will make
Microsoft the third-largest gaming company behind Tencent and Sony.
CEO Satya Nadella said the acquisition will help Microsoft compete
in the metaverse.
"""

print("Entidades encontradas:")
for entity in ner.extract(text):
    print(f"  {entity['text']:25} → {entity['label']}")

print("\nAgrupadas por tipo:")
by_type = ner.extract_by_type(text)
for label, entities in by_type.items():
    print(f"  {label}: {entities}")
```

### 3.4 NER con Transformers (BERT)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List, Dict

class BERTNERExtractor:
    """
    NER usando modelos BERT pre-entrenados.
    Más preciso que spaCy para muchas tareas.
    """

    def __init__(self, model_name: str = 'dslim/bert-base-NER'):
        """
        Args:
            model_name: Modelo NER pre-entrenado
                - dslim/bert-base-NER: General, inglés
                - Jean-Baptiste/camembert-ner: Francés
                - dbmdz/bert-large-cased-finetuned-conll03-english: CoNLL
        """
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple"  # Agrupa tokens de misma entidad
        )

    def extract(self, text: str) -> List[Dict]:
        """
        Extrae entidades con scores de confianza.
        """
        entities = self.ner_pipeline(text)

        results = []
        for ent in entities:
            results.append({
                'text': ent['word'],
                'label': ent['entity_group'],
                'score': ent['score'],
                'start': ent['start'],
                'end': ent['end'],
            })

        return results

    def extract_high_confidence(self, text: str,
                                min_score: float = 0.9) -> List[Dict]:
        """
        Extrae solo entidades con alta confianza.
        """
        entities = self.extract(text)
        return [e for e in entities if e['score'] >= min_score]


# Ejemplo
ner = BERTNERExtractor()

text = "Elon Musk announced that Tesla will open a new factory in Berlin, Germany next year."

print("Entidades (BERT NER):")
for entity in ner.extract(text):
    print(f"  {entity['text']:20} → {entity['label']:10} (conf: {entity['score']:.2f})")
```

---

## 4. NER Personalizado para Seguridad

### 4.1 Extractor de IOCs

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class IOC:
    """Indicator of Compromise."""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0

class SecurityNER:
    """
    NER especializado para extraer IOCs de textos de seguridad.

    Extrae:
    - IPs (IPv4, IPv6)
    - Dominios
    - URLs
    - Hashes (MD5, SHA1, SHA256)
    - CVEs
    - Emails
    - Rutas de archivos
    - Claves de registro
    """

    def __init__(self):
        # Patrones regex para IOCs
        self.patterns = {
            'ipv4': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'ipv6': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'domain': r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|co|info|biz|xyz|top|ru|cn|tk)\b',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'md5': r'\b[a-fA-F0-9]{32}\b',
            'sha1': r'\b[a-fA-F0-9]{40}\b',
            'sha256': r'\b[a-fA-F0-9]{64}\b',
            'cve': r'CVE-\d{4}-\d{4,7}',
            'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            'windows_path': r'[A-Za-z]:\\(?:[^\\\/:*?"<>|\r\n]+\\)*[^\\\/:*?"<>|\r\n]*',
            'unix_path': r'(?:/[^\s/]+)+/?',
            'registry': r'HKEY_[A-Z_]+(?:\\[^\s\\]+)+',
            'bitcoin': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
        }

        # Compilar patrones
        self.compiled = {
            name: re.compile(pattern)
            for name, pattern in self.patterns.items()
        }

    def extract(self, text: str) -> List[IOC]:
        """
        Extrae todos los IOCs del texto.
        """
        iocs = []

        for ioc_type, pattern in self.compiled.items():
            for match in pattern.finditer(text):
                ioc = IOC(
                    text=match.group(),
                    type=ioc_type,
                    start=match.start(),
                    end=match.end()
                )
                iocs.append(ioc)

        # Ordenar por posición
        iocs.sort(key=lambda x: x.start)

        # Eliminar duplicados y overlaps
        iocs = self._remove_overlaps(iocs)

        return iocs

    def _remove_overlaps(self, iocs: List[IOC]) -> List[IOC]:
        """
        Elimina IOCs que se solapan, manteniendo el más específico.
        """
        if not iocs:
            return []

        # Prioridad de tipos (más específico primero)
        priority = {
            'cve': 1,
            'sha256': 2,
            'sha1': 3,
            'md5': 4,
            'url': 5,
            'email': 6,
            'ipv4': 7,
            'ipv6': 8,
            'domain': 9,
            'registry': 10,
            'windows_path': 11,
            'unix_path': 12,
            'bitcoin': 13,
        }

        result = []
        for ioc in iocs:
            # Verificar si se solapa con alguno existente
            overlaps = False
            for existing in result:
                if (ioc.start < existing.end and ioc.end > existing.start):
                    # Se solapan - mantener el de mayor prioridad
                    if priority.get(ioc.type, 99) < priority.get(existing.type, 99):
                        result.remove(existing)
                        result.append(ioc)
                    overlaps = True
                    break

            if not overlaps:
                result.append(ioc)

        return result

    def extract_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Agrupa IOCs por tipo.
        """
        iocs = self.extract(text)

        by_type = {}
        for ioc in iocs:
            if ioc.type not in by_type:
                by_type[ioc.type] = []
            by_type[ioc.type].append(ioc.text)

        return by_type

    def defang(self, text: str) -> str:
        """
        "Defangs" IOCs para compartir de forma segura.
        192.168.1.1 → 192[.]168[.]1[.]1
        http://evil.com → hxxp://evil[.]com
        """
        # Defang IPs
        for match in self.compiled['ipv4'].finditer(text):
            ip = match.group()
            defanged = ip.replace('.', '[.]')
            text = text.replace(ip, defanged)

        # Defang URLs
        text = text.replace('http://', 'hxxp://')
        text = text.replace('https://', 'hxxps://')

        # Defang dominios
        for match in self.compiled['domain'].finditer(text):
            domain = match.group()
            defanged = domain.replace('.', '[.]')
            text = text.replace(domain, defanged)

        return text


# Ejemplo
extractor = SecurityNER()

threat_report = """
THREAT INTELLIGENCE REPORT

The APT group deployed malware from the C2 server at 192.168.1.100.
The payload was downloaded from http://malicious-domain.xyz/payload.exe
and had SHA256 hash: 5d41402abc4b2a76b9719d911017c592b9719d911017c5925d41402abc4b2a76

The malware exploited CVE-2021-44228 (Log4Shell) and CVE-2022-22965.
Communications were sent to evil@badactor.com.

Registry persistence: HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run

Affected file: C:\\Windows\\System32\\malware.dll
"""

print("IOCs Extraídos:")
print("-" * 60)
for ioc in extractor.extract(threat_report):
    print(f"  [{ioc.type:15}] {ioc.text}")

print("\nAgrupados por tipo:")
by_type = extractor.extract_by_type(threat_report)
for ioc_type, values in by_type.items():
    print(f"  {ioc_type}: {values}")

print("\nTexto defanged:")
print(extractor.defang(threat_report[:200]))
```

### 4.2 NER con Modelo Custom (Fine-tuning)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments
)
from typing import List, Dict, Tuple
import numpy as np

class SecurityNERDataset(Dataset):
    """
    Dataset para entrenar NER de seguridad.
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[List[str]],
                 tokenizer,
                 label2id: Dict[str, int],
                 max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]

        # Tokenizar
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Alinear labels con tokens
        word_ids = encoding.word_ids()
        label_ids = []

        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignorar tokens especiales
            elif word_id != previous_word_id:
                label_ids.append(self.label2id[word_labels[word_id]])
            else:
                # Para subtokens, usar I- en lugar de B-
                label = word_labels[word_id]
                if label.startswith('B-'):
                    label = 'I-' + label[2:]
                label_ids.append(self.label2id.get(label, self.label2id['O']))

            previous_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids)
        }


def train_security_ner():
    """
    Entrena un modelo NER para IOCs de seguridad.
    """
    # Datos de ejemplo (en práctica, usar dataset anotado)
    texts = [
        ["The", "attacker", "used", "192.168.1.100", "as", "C2"],
        ["Malware", "hash", ":", "5d41402abc4b2a76b9719d911017c592"],
        ["Exploited", "CVE-2021-44228", "vulnerability"],
    ]

    labels = [
        ["O", "O", "O", "B-IP", "O", "O"],
        ["O", "O", "O", "B-HASH"],
        ["O", "B-CVE", "O"],
    ]

    # Etiquetas
    label_list = ['O', 'B-IP', 'I-IP', 'B-HASH', 'I-HASH',
                  'B-CVE', 'I-CVE', 'B-DOMAIN', 'I-DOMAIN',
                  'B-URL', 'I-URL', 'B-MALWARE', 'I-MALWARE']

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # Tokenizer y modelo
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # Dataset
    dataset = SecurityNERDataset(texts, labels, tokenizer, label2id)

    # Training args
    training_args = TrainingArguments(
        output_dir='./security_ner',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Entrenar
    trainer.train()

    return model, tokenizer, id2label


# model, tokenizer, id2label = train_security_ner()
```

---

## 5. Evaluación de NER

### 5.1 Métricas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MÉTRICAS PARA NER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EVALUACIÓN A NIVEL DE TOKEN:                                               │
│    Precision = TP / (TP + FP)                                               │
│    Recall = TP / (TP + FN)                                                  │
│    F1 = 2 * (Precision * Recall) / (Precision + Recall)                     │
│                                                                              │
│  EVALUACIÓN A NIVEL DE ENTIDAD (más estricta):                              │
│    Una entidad es correcta SOLO si:                                         │
│    - Todos los tokens están correctamente etiquetados                       │
│    - Los límites (inicio/fin) son exactos                                   │
│    - El tipo es correcto                                                    │
│                                                                              │
│  EJEMPLO:                                                                    │
│                                                                              │
│  Gold:   "New York City" → B-LOC I-LOC I-LOC                                │
│  Pred1:  "New York City" → B-LOC I-LOC I-LOC  ✓ Correcto                   │
│  Pred2:  "New York City" → B-LOC I-LOC O      ✗ Parcial (límite incorrecto)│
│  Pred3:  "New York City" → B-ORG I-ORG I-ORG  ✗ Tipo incorrecto            │
│                                                                              │
│  PARTIAL MATCHING (relajado):                                                │
│    - Exact: límites y tipo exactos                                          │
│    - Partial: overlap > 0 y tipo correcto                                   │
│    - Type: tipo correcto, límites pueden variar                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementación de Evaluación

```python
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from typing import List

def evaluate_ner(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> dict:
    """
    Evalúa predicciones NER usando seqeval.

    Args:
        true_labels: Labels reales en formato BIO
        pred_labels: Labels predichos en formato BIO

    Returns:
        Diccionario con métricas
    """
    # Métricas generales
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # Reporte por clase
    report = classification_report(true_labels, pred_labels)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }


# Ejemplo
true_labels = [
    ['O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'O', 'B-DATE', 'I-DATE'],
]

pred_labels = [
    ['O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O'],  # Perfecto
    ['B-LOC', 'O', 'O', 'O', 'B-DATE', 'I-DATE'],  # Error en LOC
]

metrics = evaluate_ner(true_labels, pred_labels)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
print("\nReporte por clase:")
print(metrics['report'])
```

---

## 6. Aplicaciones en Seguridad

### 6.1 Extracción de Threat Intelligence

```python
import spacy
from typing import Dict, List
import re

class ThreatIntelExtractor:
    """
    Extractor de inteligencia de amenazas de textos.

    Extrae:
    - Actores de amenazas (APT groups)
    - Malware
    - Técnicas/tácticas
    - IOCs
    - Sectores afectados
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

        # Patrones para entidades de seguridad
        self.apt_pattern = re.compile(
            r'\b(APT\d+|Lazarus|Fancy Bear|Cozy Bear|Equation Group|'
            r'Turla|Sandworm|Charming Kitten|Stone Panda)\b',
            re.IGNORECASE
        )

        self.malware_pattern = re.compile(
            r'\b(WannaCry|Emotet|TrickBot|Ryuk|REvil|Conti|'
            r'Cobalt Strike|Mimikatz|NotPetya|Stuxnet)\b',
            re.IGNORECASE
        )

        self.technique_pattern = re.compile(
            r'\b(phishing|spear phishing|ransomware|supply chain|'
            r'watering hole|credential stuffing|brute force|'
            r'sql injection|buffer overflow|zero day|lateral movement)\b',
            re.IGNORECASE
        )

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae toda la inteligencia de amenazas del texto.
        """
        doc = self.nlp(text)

        intel = {
            'threat_actors': [],
            'malware': [],
            'techniques': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'iocs': {
                'ips': [],
                'domains': [],
                'hashes': [],
                'cves': [],
            }
        }

        # APT groups y malware (regex)
        intel['threat_actors'] = list(set(self.apt_pattern.findall(text)))
        intel['malware'] = list(set(self.malware_pattern.findall(text)))
        intel['techniques'] = list(set(self.technique_pattern.findall(text)))

        # Entidades de spaCy
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                intel['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                intel['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                intel['dates'].append(ent.text)

        # IOCs (regex)
        intel['iocs']['ips'] = re.findall(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text
        )
        intel['iocs']['cves'] = re.findall(
            r'CVE-\d{4}-\d{4,7}', text
        )
        intel['iocs']['hashes'] = re.findall(
            r'\b[a-fA-F0-9]{32,64}\b', text
        )

        # Deduplicar
        for key in ['organizations', 'locations', 'dates']:
            intel[key] = list(set(intel[key]))

        for key in intel['iocs']:
            intel['iocs'][key] = list(set(intel['iocs'][key]))

        return intel


# Ejemplo
extractor = ThreatIntelExtractor()

threat_report = """
APT29, also known as Cozy Bear, has been observed targeting government
organizations in the United States and European Union since January 2024.

The group is using a new variant of the Cobalt Strike framework to deploy
TrickBot malware. The campaign exploits CVE-2023-12345 and uses phishing
emails as the initial attack vector.

C2 servers have been identified at 45.33.32.156 and 203.0.113.50.
The malware hash is 5d41402abc4b2a76b9719d911017c592.

Researchers at Microsoft and Mandiant have attributed this campaign
to Russian state-sponsored actors.
"""

intel = extractor.extract(threat_report)

print("=== THREAT INTELLIGENCE EXTRAÍDA ===\n")
for category, values in intel.items():
    if isinstance(values, dict):
        print(f"{category}:")
        for subcategory, subvalues in values.items():
            if subvalues:
                print(f"  {subcategory}: {subvalues}")
    elif values:
        print(f"{category}: {values}")
```

---

## 7. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NER Y POS TAGGING - RESUMEN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POS TAGGING:                                                                │
│    • Asigna categoría gramatical a cada palabra                             │
│    • Útil para: parsing, desambiguación, preprocessing                      │
│    • Herramientas: NLTK, spaCy                                              │
│                                                                              │
│  NER:                                                                        │
│    • Identifica entidades nombradas (personas, lugares, etc.)               │
│    • Esquema BIO: B=inicio, I=continuación, O=fuera                         │
│    • Herramientas: spaCy, Transformers (BERT-NER)                           │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • IOCs: IPs, hashes, CVEs, dominios, URLs                                │
│    • Threat actors: APT groups, malware names                               │
│    • Técnicas: phishing, ransomware, injection                              │
│    • Combinar regex + NER estadístico para mejor cobertura                  │
│                                                                              │
│  EVALUACIÓN:                                                                 │
│    • Métricas a nivel de entidad (estricto)                                 │
│    • F1-score por tipo de entidad                                           │
│    • Librería: seqeval                                                       │
│                                                                              │
│  PIPELINE TÍPICO:                                                            │
│    1. spaCy para entidades genéricas (ORG, LOC, etc.)                       │
│    2. Regex para IOCs específicos (IPs, hashes, CVEs)                       │
│    3. BERT NER fine-tuned para entidades de dominio específico              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
