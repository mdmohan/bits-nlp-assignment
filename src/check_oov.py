from lm.data import load_corpus_files, Vocab

corpus = load_corpus_files(
    "data/processed/webtext/test.txt",
    "data/processed/webtext/valid.txt",
    "data/processed/webtext/train.txt",
).lowercase()

vocab = Vocab(corpus.train, min_freq=1)  # matches webtext.yaml
print(f"Validation OOV rate: {vocab.oov_rate(corpus.valid):.2%}")