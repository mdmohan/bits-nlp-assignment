from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams

tokenizer = TreebankWordTokenizer()

# Learn the vocabulary from training data only.
train_tokens = [tokenizer.tokenize(s.lower()) for s in train_sentences]
test_tokens = [tokenizer.tokenize(s.lower()) for s in test_sentences]

word_counts = Counter(word for sentence in train_tokens for word in sentence)
vocab = {word for word, count in word_counts.items() if count >= 2}

def map_rare_words(tokenized_sentences):
    return [
        [word if word in vocab else "<UNK>" for word in sentence]
        for sentence in tokenized_sentences
    ]

train_tokens = map_rare_words(train_tokens)
test_tokens = map_rare_words(test_tokens)

def count_ngrams(tokenized_sentences, n):
    counts = Counter()
    for sentence in tokenized_sentences:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        counts.update(ngrams(padded, n))
    return counts

for n in (2, 3):
    train_counts = count_ngrams(train_tokens, n)
    test_counts = count_ngrams(test_tokens, n)

    candidates = [
        (ngram, train_counts[ngram], test_count)
        for ngram, test_count in test_counts.items()
        if train_counts[ngram] <= 2
    ]
    candidates.sort(key=lambda item: (item[1], -item[2]))

    print(f"\n{n}-grams: first 10 unseen or rare test sequences")
    for ngram, train_count, test_count in candidates[:10]:
        print(f"{' '.join(ngram)} | train={train_count}, test={test_count}")