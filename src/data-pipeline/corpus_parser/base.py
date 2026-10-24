import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union


try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

    
try:
    from typing import override
except ImportError:
    from typing_extensions import override


class BaseParser(ABC):
    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, corpus_name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[corpus_name.lower()] = cls

    @classmethod
    def create(cls, corpus_name: str, **kwargs) -> "BaseParser":
        key = corpus_name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown corpus '{corpus_name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[key](**kwargs)

    @abstractmethod
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        pass

    @staticmethod
    def read_text_robust(path: Path) -> str:
        raw_bytes = path.read_bytes()
        for encoding in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1'):
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw_bytes.decode('utf-8', errors='replace')

    @classmethod
    def split_sentences(cls, text: str):
        """Splits paragraph into sentences using NLTK or regex fallback."""
        if HAS_NLTK:
            return sent_tokenize(text)
        return [s.strip() for s in re.split(cls.SENTENCE_PATTERN, text) if s.strip()]


     

#

# ==========================================
# 2. PENN TREEBANK (PTB)
# ==========================================
def parse_ptb_file(filepath):
    """
    Parses Penn Treebank parsed/mrg files (e.g., 'wsj_0001.mrg').
    Extracts sentences directly from S-expression parse trees (S ...).
    """
    from nltk.tree import Tree

    raw_text = Path(filepath).read_text(encoding='utf-8', errors='ignore')

    # Find top-level s-expressions enclosed in parentheses
    tree_strings = re.findall(r'\(\s*S[^\n]*\n(?:\s+.*\n)*?\s*\)', raw_text)

    # If file format uses flat single-line tree expressions
    if not tree_strings:
        tree_strings = re.findall(r'\(\s*S[\s\S]*?\n\n|\(\s*S[\s\S]*?\)\s*$', raw_text)

    sentences = []
    for t_str in tree_strings:
        try:
            tree = Tree.fromstring(t_str)
            # Flatten tree leaves into a cleaned sentence string
            tokens = [leaf for leaf in tree.leaves() if leaf not in ('-NONE-', '*U*', '*T*')]
            sentence = ' '.join(tokens)
            # Basic cleanup for punctuation spaces (e.g., " . " -> ".")
            sentence = re.sub(r'\s+([,.\?\!\'\"])', r'\1', sentence)
            if sentence:
                sentences.append(sentence)
        except Exception:
            continue

    # PTB files do not explicitly group paragraphs, so we structure sentences directly
    return [{
        'paragraph_number': 1,
        'paragraph_text': ' '.join(sentences),
        'sentences': sentences
    }]


# ==========================================
# 4. WEBTEXT / OPENWEBTEXT CORPUS
# ==========================================
def parse_webtext_file(filepath):
    """
    Parses WebText/OpenWebText files (handles JSONL, JSON, or raw web-scraped text).
    Unescapes HTML entities and normalizes web linebreaks.
    """
    raw_text = Path(filepath).read_text(encoding='utf-8', errors='ignore')
    documents = []

    # Case A: JSON Lines (JSONL) format (standard for OpenWebText)
    if filepath.endswith('.jsonl') or (raw_text.startswith('{') and '\n{' in raw_text):
        for line in raw_text.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                documents.append(data.get('text', ''))
    # Case B: Standard text web-dump file
    else:
        documents.append(raw_text)

    results = []
    for doc_idx, doc in enumerate(documents, start=1):
        # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
        clean_doc = html.unescape(doc)

        # Normalize line endings
        clean_doc = clean_doc.replace('\r\n', '\n')

        # Split by double newline for web paragraphs
        paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', clean_doc) if p.strip()]

        doc_paragraphs = []
        for p_idx, paragraph in enumerate(paragraphs, start=1):
            doc_paragraphs.append({
                'paragraph_number': p_idx,
                'sentences': sent_tokenize(paragraph)
            })

        results.append({
            'document_id': doc_idx,
            'paragraphs': doc_paragraphs
        })

    return results

    