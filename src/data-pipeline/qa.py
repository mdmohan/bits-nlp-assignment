import nltk
from nltk.corpus import names

# Ensure required corpus data is downloaded
nltk.download("names", quiet=True)

# 1. Check available file IDs in the corpus
file_ids = names.fileids()
print("Corpus files:", file_ids)

# 2. Check counts to evaluate balance/size
female_names = names.words("female.txt")
male_names = names.words("male.txt")

print(f"Number of female names: {len(female_names)}")
print(f"Number of male names: {len(male_names)}")

# 3. Basic Quality Check: Check for overlap (duplicates across categories)
overlap = set(female_names).intersection(set(male_names))
print(f"Names appearing in both lists (potential ambiguity): {len(overlap)}")
