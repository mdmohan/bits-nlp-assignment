import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    from typing import override
except ImportError:
    from typing_extensions import override

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from .base import BaseParser, override


class GeneralParser(BaseParser, corpus_name='general'):
    """
    Universal/Corpus-agnostic parser that automatically inspects input content 
    (Plain Text, Markdown, HTML/XML/SGML, JSON, or JSONL) and extracts structured 
    paragraphs and sentences into the standard output format.
    """

    @override
    def parse(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(filepath)
        raw_text = self.read_text_robust(path)

        if not raw_text.strip():
            return []

        suffix = path.suffix.lower()

        # 1. Dynamically route content based on extension or text structure
        if suffix in ('.json', '.jsonl') or self._is_json(raw_text):
            paragraphs_raw = self._parse_json(raw_text)
        elif suffix in ('.html', '.htm', '.xml', '.sgm', '.sgml') or self._is_markup(raw_text):
            paragraphs_raw = self._parse_markup(raw_text)
        elif suffix in ('.md', '.markdown'):
            paragraphs_raw = self._parse_markdown(raw_text)
        else:
            paragraphs_raw = self._parse_plaintext(raw_text)

        # 2. Assemble uniform schema: List[Dict[str, Any]]
        result = []
        p_idx = 1
        for paragraph in paragraphs_raw:
            sentences = self.split_sentences(paragraph)
            if sentences:
                result.append({
                    'paragraph_number': p_idx,
                    'paragraph_text': paragraph,
                    'sentences': sentences
                })
                p_idx += 1

        return result

    def _is_json(self, text: str) -> bool:
        s = text.strip()
        return (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']'))

    def _is_markup(self, text: str) -> bool:
        s = text.strip().lower()
        return s.startswith('<') and '>' in s and ('</' in s or '/>' in s or '<body' in s or '<html' in s or '<p' in s)

    def _parse_plaintext(self, text: str) -> List[str]:
        text = text.replace('\r\n', '\n')
        paragraphs = [
            re.sub(r'\s+', ' ', p).strip() 
            for p in re.split(r'\n\s*\n', text) 
            if p.strip()
        ]
        return paragraphs if paragraphs else ([text.strip()] if text.strip() else [])

    def _parse_markdown(self, text: str) -> List[str]:
        # Strip fenced code blocks, image links, and header tokens
        clean = re.sub(r'```[\s\S]*?```', '', text)
        clean = re.sub(r'!\[.*?\]\(.*?\)', '', clean)
        clean = re.sub(r'#+\s*', '', clean)
        return self._parse_plaintext(clean)

    def _parse_markup(self, text: str) -> List[str]:
        if HAS_BS4:
            soup = BeautifulSoup(text, 'html.parser')
            # Pull text from structural tags first
            blocks = soup.find_all(['p', 'div', 'article', 'section', 'body'])
            if blocks:
                candidates = [re.sub(r'\s+', ' ', b.get_text()).strip() for b in blocks if b.get_text().strip()]
                # Filter duplicate text nested inside parent elements
                unique_paras = []
                for cand in candidates:
                    if cand and not any(cand in other and cand != other for other in candidates):
                        unique_paras.append(cand)
                if unique_paras:
                    return unique_paras
            return self._parse_plaintext(soup.get_text())
        else:
            # Fallback regex tag stripping if BeautifulSoup is unavailable
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            return self._parse_plaintext(clean_text)

    def _parse_json(self, text: str) -> List[str]:
        extracted_strings = []
        lines = text.strip().splitlines()
        
        # Check for JSON Lines (.jsonl)
        is_jsonl = len(lines) > 1 and all(l.strip().startswith('{') for l in lines if l.strip())

        if is_jsonl:
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        extracted_strings.extend(self._recursive_json_extract(obj))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                obj = json.loads(text)
                extracted_strings.extend(self._recursive_json_extract(obj))
            except json.JSONDecodeError:
                return self._parse_plaintext(text)

        all_paragraphs = []
        for string_val in extracted_strings:
            all_paragraphs.extend(self._parse_plaintext(string_val))
        return all_paragraphs

    def _recursive_json_extract(self, obj: Any) -> List[str]:
        """Recursively extracts string fields from nested dictionaries and lists."""
        results = []
        if isinstance(obj, str):
            if obj.strip():
                results.append(obj)
        elif isinstance(obj, dict):
            # Prioritize standard text fields if present
            text_keys = ('text', 'body', 'content', 'paragraph', 'paragraphs', 'article', 'sentence', 'sentences')
            found = False
            for key in text_keys:
                if key in obj:
                    found = True
                    results.extend(self._recursive_json_extract(obj[key]))
            if not found:
                for val in obj.values():
                    results.extend(self._recursive_json_extract(val))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._recursive_json_extract(item))
        return results