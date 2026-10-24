from .base import BaseParser
from . import gutenberg   # Triggers subclass registration hooks
from . import brown
from . import ptb
from . import reuters
from . import generic

def create_parser(corpus_name: str, **kwargs) -> BaseParser:
    return BaseParser.create(corpus_name, **kwargs)

__all__ = ["BaseParser", "create_parser"]