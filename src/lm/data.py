"""Load pipeline split files and prepare them for the neural language models."""

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

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


def load_corpus_files(
    train: str | Path,
    valid: str | Path,
    test: str | Path,
    train_percent: float = 100,
) -> Corpus:
    """Read the split files, optionally using the first percentage of train sentences."""
    if not 0 < train_percent <= 100:
        raise ValueError("train_percent must be greater than 0 and at most 100")

    def read(path: str | Path, limit: int | None = None) -> list[Sentence]:
        with Path(path).open(encoding="utf-8") as handle:
            sentences = []
            for line in handle:
                if line.strip():
                    sentences.append(line.split())
                    if limit is not None and len(sentences) >= limit:
                        break
            return sentences

    train_path = Path(train)
    train_limit = None
    if train_percent < 100:
        with train_path.open(encoding="utf-8") as handle:
            total_train = sum(1 for line in handle if line.strip())
        train_limit = math.ceil(total_train * train_percent / 100)
        print(
            f"train subset: using first {train_percent:g}% "
            f"({train_limit:,}/{total_train:,} sentences)"
        )

    return Corpus(read(train_path, train_limit), read(valid), read(test))


# ---------------------------------------------------------------- vocabulary


class Vocab:
    """Word <-> id mapping built from training sentences only. Rare words map to <unk>."""

    def __init__(self, sentences: list[Sentence], min_freq: int = 2, max_size: int | None = None):
        counts = Counter(word for sentence in sentences for word in sentence)
        self._initialize(counts, min_freq, max_size)

    @classmethod
    def from_counts(cls, counts: Counter, min_freq: int = 2, max_size: int | None = None):
        """Build the same vocabulary from streaming token counts."""
        vocab = cls.__new__(cls)
        vocab._initialize(counts, min_freq, max_size)
        return vocab

    def _initialize(self, counts: Counter, min_freq: int, max_size: int | None):
        words = [word for word, count in counts.most_common(max_size) if count >= min_freq]
        self.itos = [PAD, UNK, BOS, EOS] + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

        train_tokens = sum(counts.values())
        retained_tokens = sum(counts[word] for word in words)
        print(
            "vocab: "
            f"train_tokens={train_tokens:,} "
            f"train_types={len(counts):,} "
            f"retained_words={len(words):,} "
            f"vocab_size={len(self):,} (includes {len((PAD, UNK, BOS, EOS))} special tokens) "
            f"min_freq={min_freq} max_size={'unlimited' if max_size is None else max_size} "
            f"train_tokens_to_unk={train_tokens - retained_tokens:,}"
        )

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


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar lengths and cap padded sequence positions per batch."""

    def __init__(self, dataset: SentenceDataset, batch_size: int, shuffle: bool,
                 max_padded_tokens: int | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.lengths = [len(item) for item in dataset.items]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_padded_tokens = max_padded_tokens or batch_size * 64

    def _batches(self) -> list[list[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = []
        batch = []
        batch_max_length = 0
        for index in indices:
            proposed_max_length = max(batch_max_length, self.lengths[index])
            proposed_size = len(batch) + 1
            exceeds_token_budget = proposed_size * proposed_max_length > self.max_padded_tokens
            if batch and (proposed_size > self.batch_size or exceeds_token_budget):
                batches.append(batch)
                batch = []
                batch_max_length = 0
            batch.append(index)
            batch_max_length = max(batch_max_length, self.lengths[index])
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()

        if self.shuffle:
            random.shuffle(batches)
            for batch in batches:
                random.shuffle(batch)
        yield from batches

    def __len__(self):
        return len(self._batches())


def make_loader(sentences: list[Sentence], vocab: Vocab, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SentenceDataset(sentences, vocab)

    def collate(batch):
        inputs, targets = zip(*batch)
        return (
            pad_sequence(inputs, batch_first=True, padding_value=vocab.pad_id),
            pad_sequence(targets, batch_first=True, padding_value=vocab.pad_id),
        )

    return DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(dataset, batch_size, shuffle),
        collate_fn=collate,
    )
