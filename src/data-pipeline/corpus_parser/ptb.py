import re
from pathlib import Path
from typing import Any, Dict, Union
from .base import BaseParser, override

class PtbParser(BaseParser, corpus_name='ptb'):
    def __init__(self, strip_headers: bool = True):
        self.strip_headers = strip_headers

    @override
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
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

