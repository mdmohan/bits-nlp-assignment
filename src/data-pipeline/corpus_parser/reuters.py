import re
from pathlib import Path
from typing import Any, Dict, List, Union
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from .base import BaseParser, override

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

    
class ReutersParser(BaseParser, corpus_name='reuters'):

    @override
    def parse(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        raw_text = self.read_text_robust(filepath)
        # Case A: Raw SGML format with <reuters> tags (e.g. reut2-000.sgm)
        if '<reuters' in raw_text.lower():
            if not HAS_BS4:
                raise ImportError("beautifulsoup4 is required for SGML Reuters parsing (`pip install beautifulsoup4`).")

            soup = BeautifulSoup(raw_text, 'html.parser')
            articles = soup.find_all('reuters')

            result = []
            idx = 1

            for article in articles:
                body = article.body.text.strip() if article.body else ""
                if not body:
                    continue

                body = re.sub(r'Reuter\s*&#3;?$', '', body, flags=re.IGNORECASE).strip()
                paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', body) if p.strip()]

                for paragraph in paragraphs:
                    sentences = self.split_sentences(paragraph)
                    result.append({
                        'paragraph_number': idx,
                        'paragraph_text': paragraph,
                        'sentences': sentences
                    })
                    idx += 1
            return result

        # Case B: Plain-text files (e.g. NLTK corpora files like 14826)
        content = raw_text.replace('\r\n', '\n').strip()
        if not content:
            return []

        paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', content) if p.strip()]
        if not paragraphs:
            paragraphs = [content.replace('\n', ' ')]

        result = []
        for idx, paragraph in enumerate(paragraphs, start=1):
            sentences = self.split_sentences(paragraph)
            result.append({
                'paragraph_number': idx,
                'paragraph_text': paragraph,
                'sentences': sentences
            })

        return result
