"""Shared, extensible tokenization for corpus loading and model evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer


class Tokenizer(ABC):
    """Interface implemented by each named tokenizer."""

    name: str
    aliases: tuple[str, ...] = ()

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split one sentence or prompt into tokens."""


class WhitespaceTokenizer(Tokenizer):
    name = "space"
    aliases = ("whitespace", "ws")

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class WordPunctTokenizerAdapter(Tokenizer):
    name = "wp"
    aliases = ("wordpunct", "word-punct")

    def __init__(self):
        self._tokenizer = WordPunctTokenizer()

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)


class TreebankTokenizer(Tokenizer):
    name = "tb"
    aliases = ("treebank",)

    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)


_TOKENIZERS: dict[str, Tokenizer] = {}
_ALIASES: dict[str, str] = {}


def register_tokenizer(tokenizer_class: type[Tokenizer]) -> None:
    """Register a tokenizer class by its canonical name and aliases."""
    tokenizer = tokenizer_class()
    names = (tokenizer.name, *tokenizer.aliases)
    for name in names:
        key = name.strip().casefold()
        if not key or key in _ALIASES:
            raise ValueError(f"Tokenizer name or alias is empty or already registered: {name!r}")
        _ALIASES[key] = tokenizer.name
    _TOKENIZERS[tokenizer.name] = tokenizer


def tokenizer_names(include_aliases: bool = True) -> tuple[str, ...]:
    """Return accepted CLI/config names, with short names first."""
    if include_aliases:
        return tuple(_ALIASES)
    return tuple(_TOKENIZERS)


def canonical_tokenizer_name(name: str | None) -> str:
    """Resolve a CLI/config name to its stable canonical identifier."""
    requested = "space" if name is None else name.strip().casefold()
    try:
        return _ALIASES[requested]
    except KeyError as exc:
        choices = ", ".join(_TOKENIZERS)
        raise ValueError(f"Unknown tokenizer {name!r}; choose one of: {choices}") from exc


def get_tokenizer(name: str | None = "space") -> Tokenizer:
    """Return the registered tokenizer instance for a short name or alias."""
    return _TOKENIZERS[canonical_tokenizer_name(name)]


def tokenize(text: str, name: str | None = "space", lowercase: bool = False) -> list[str]:
    """Tokenize text and optionally lowercase tokens using the shared registry."""
    words = get_tokenizer(name).tokenize(text)
    return [word.lower() for word in words] if lowercase else words


register_tokenizer(WhitespaceTokenizer)
register_tokenizer(WordPunctTokenizerAdapter)
register_tokenizer(TreebankTokenizer)
