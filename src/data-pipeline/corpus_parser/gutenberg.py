import re
from pathlib import Path
from typing import Any, Dict, Union
from .base import BaseParser, override

class GutenbergParser(BaseParser, corpus_name='gutenberg'):
    def __init__(self, strip_headers: bool = True):
        self.strip_headers = strip_headers

    @override
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        path = Path(filepath)
        content = self.read_text_robust(path)

        if self.strip_headers:
            start_match = re.search(r'\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*', content, re.I)
            end_match = re.search(r'\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*', content, re.I)
            start_idx = start_match.end() if start_match else 0
            end_idx = end_match.start() if end_match else len(content)
            content = content[start_idx:end_idx].strip()

        content = content.replace('\r\n', '\n')
        paragraphs_raw = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'

        parsed_paragraphs = []
        for index, paragraph in enumerate(paragraphs_raw, start=1):
            sentences = [s.strip() for s in re.split(sentence_pattern, paragraph) if s.strip()]
            parsed_paragraphs.append({
                'paragraph_number': index,
                'paragraph_text': paragraph,
                'sentences': sentences,
            })
        print (filepath, len(parsed_paragraphs))
        return parsed_paragraphs