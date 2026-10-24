# ==========================================
# 1. BROWN CORPUS
# ==========================================

import re
from pathlib import Path
from typing import Any, Dict, Union
from .base import BaseParser, override
import nltk
from nltk.tokenize import sent_tokenize

class BrownParser(BaseParser, corpus_name = 'brown'):
    def __init__(self, strip_headers: bool = True):
        self.strip_headers = strip_headers

    @override
    def parse(self, filepath, strip_pos_tags=True):
        """
        Parses raw Brown Corpus files (e.g., 'ca01').
        Raw Brown files use POS tagging formatted as 'word/tag' (e.g., 'The/at Fulton/np-tl').
        """
        raw_text = Path(filepath).read_text(encoding='utf-8', errors='ignore')
    
        # Strip POS tags if present (word/tag -> word)
        if strip_pos_tags:
            raw_text = re.sub(r'/[\w$\-\+\*]+', '', raw_text)
    
        # Paragraphs in Brown raw files are separated by blank lines or tabs/indentation
        paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
    
        result = []
        for idx, paragraph in enumerate(paragraphs, start=1):
            sentences = sent_tokenize(paragraph)
            result.append({
                'paragraph_number': idx,
                'paragraph_text': paragraph,
                'sentences': sentences
            })
            
       
        return result
