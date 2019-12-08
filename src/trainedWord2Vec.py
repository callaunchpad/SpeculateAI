import glob
import json
from Pipeline import *

unique_words = set()
unique_words.add("START")
unique_words.add("END")
unique_words.add("PAD")
unique_words.add("UNK")

json_files = glob.glob("../data/small_financial_news/*.json")

for json_file in json_files:
    # Open the json file and pull the headline
    with open(json_file, encoding='utf-8') as file:
        # Load the title string into headline
        headline = json.loads(file.read())['title']
        # Clean the headline
        headline = clean_headline(headline)
        # Add it to the list of headlines

        for word in headline.split(" "):
            unique_words.add(word)

# Save the vocabulary
with open("vocabulary.txt", 'w') as vocab_file:
    for word in unique_words:
        vocab_file.write(f"{word}\n")