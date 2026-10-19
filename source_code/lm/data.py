"""Turn any text dataset into tokenised sentences, a vocabulary and PyTorch batches.

Every loader returns a `Corpus`: three lists of sentences, where a sentence is a list of
word tokens. Everything downstream (vocabulary, models, training) only sees a `Corpus`,
so adding a new dataset means writing one loader function and registering it in LOADERS.
"""

import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"

Sentence = list[str]


@dataclass
class Corpus:
    train: list[Sentence]
    valid: list[Sentence]
    test: list[Sentence]

    def lowercase(self) -> "Corpus":
        def lower(sentences):
            return [[word.lower() for word in sentence] for sentence in sentences]

        return Corpus(lower(self.train), lower(self.valid), lower(self.test))

    def describe(self) -> dict:
        stats = {}
        for split, sentences in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            stats[f"{split}_sentences"] = len(sentences)
            stats[f"{split}_tokens"] = sum(len(sentence) for sentence in sentences)
        return stats


def split_documents(documents: list[list[Sentence]], ratios=(0.8, 0.1, 0.1), seed=42) -> Corpus:
    """Shuffle whole documents with a fixed seed, so no document appears in two splits."""
    documents = list(documents)
    random.Random(seed).shuffle(documents)

    n_train = int(ratios[0] * len(documents))
    n_valid = int(ratios[1] * len(documents))

    def flatten(docs):
        return [sentence for doc in docs for sentence in doc if sentence]

    return Corpus(
        train=flatten(documents[:n_train]),
        valid=flatten(documents[n_train:n_train + n_valid]),
        test=flatten(documents[n_train + n_valid:]),
    )


# ---------------------------------------------------------------- dataset loaders


def load_brown(categories=None, ratios=(0.8, 0.1, 0.1), split_seed=42) -> Corpus:
    """NLTK Brown corpus; each Brown file is one document. `categories=None` uses all genres."""
    import nltk
    from nltk.corpus import brown

    nltk.download("brown", quiet=True)
    documents = [
        [list(sentence) for sentence in brown.sents(fileid)]
        for fileid in brown.fileids(categories=categories)
    ]
    return split_documents(documents, ratios, split_seed)


def load_text_files(train: str, valid: str, test: str) -> Corpus:
    """Plain text files, one sentence per line, already split. Tokens are whitespace-separated."""

    def read(path):
        with open(path, encoding="utf-8") as handle:
            return [line.split() for line in handle if line.strip()]

    return Corpus(read(train), read(valid), read(test))


def load_huggingface(name: str, config=None, text_field="text",
                     splits=("train", "validation", "test")) -> Corpus:
    """Any Hugging Face dataset with a text column, e.g. name='Salesforce/wikitext',
    config='wikitext-2-raw-v1'. Each non-empty line becomes one whitespace-tokenised sentence."""
    from datasets import load_dataset

    dataset = load_dataset(name, config)

    def read(split):
        return [row[text_field].split() for row in dataset[split] if row[text_field].strip()]

    train, valid, test = (read(split) for split in splits)
    return Corpus(train, valid, test)


LOADERS = {
    "brown": load_brown,
    "text_files": load_text_files,
    "huggingface": load_huggingface,
}


def load_corpus(loader: str, lowercase: bool = True, **params) -> Corpus:
    corpus = LOADERS[loader](**params)
    if lowercase:
        corpus = corpus.lowercase()
    return corpus


# ---------------------------------------------------------------- vocabulary


class Vocab:
    """Word <-> id mapping built from training sentences only. Rare words map to <unk>."""

    def __init__(self, sentences: list[Sentence], min_freq: int = 2, max_size: int | None = None):
        counts = Counter(word for sentence in sentences for word in sentence)
        words = [word for word, count in counts.most_common(max_size) if count >= min_freq]
        self.itos = [PAD, UNK, BOS, EOS] + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __contains__(self, word):
        return word in self.stoi

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def encode(self, sentence: Sentence) -> list[int]:
        """'<s> w1 ... wn </s>' as ids; unknown words become <unk>."""
        ids = [self.stoi.get(word, self.unk_id) for word in sentence]
        return [self.stoi[BOS]] + ids + [self.stoi[EOS]]

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids]

    def oov_rate(self, sentences: list[Sentence]) -> float:
        """Fraction of word tokens that are not in the vocabulary."""
        tokens = [word for sentence in sentences for word in sentence]
        if not tokens:
            return 0.0
        return sum(word not in self.stoi for word in tokens) / len(tokens)


# ---------------------------------------------------------------- batching


class SentenceDataset(Dataset):
    """Each item is (input, target): '<s> w1 .. wn' predicts 'w1 .. wn </s>'."""

    def __init__(self, sentences: list[Sentence], vocab: Vocab):
        self.items = [torch.tensor(vocab.encode(sentence)) for sentence in sentences if sentence]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ids = self.items[index]
        return ids[:-1], ids[1:]


def make_loader(sentences: list[Sentence], vocab: Vocab, batch_size: int, shuffle: bool) -> DataLoader:
    def collate(batch):
        inputs, targets = zip(*batch)
        return (
            pad_sequence(inputs, batch_first=True, padding_value=vocab.pad_id),
            pad_sequence(targets, batch_first=True, padding_value=vocab.pad_id),
        )

    return DataLoader(SentenceDataset(sentences, vocab), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
