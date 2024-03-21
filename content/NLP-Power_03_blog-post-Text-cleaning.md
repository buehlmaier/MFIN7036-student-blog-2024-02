---
Title: Text Cleaning (by Group "NLP Power")
Date: 2024-3-10
Category: Progress Report
Tags: Group NLP Power
---



## Text Cleaning

We started by importing some of the necessary libraries
```python
# Import required libraries
import math
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')
```
### Downloading some necessary NLTK resources
We are downloading some necessary NLTK resources that are crucial for text processing:
```python
# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

punkt: This is a tokenizer from NLTK used to split text into sentences and words.

stopwords: This is a list of stopwords from NLTK. Stopwords are frequently occurring words in a text that do not contribute much to the overall meaning, such as “the”, “is”, “in”, etc., in English, and corresponding words in other languages.

wordnet: This is an English lexical database primarily used for lemmatization. Lemmatization is the process of converting a word to its base form (or root form), for example, converting “running” to “run”.
omw-1.4: This is an extension of WordNet that includes lexical information for multiple languages.

These resources will be used in the subsequent text processing steps, including tokenization, removal of stopwords, lemmatization, etc. These steps are crucial for understanding and analyzing text data.


This line of code loads a list of English stopwords from the NLTK library and stores them in the stop_words variable. Stopwords are popular words that contain little significance and are frequently eliminated from texts. They contain terms like "the", "is", "at", "which", and "on". Converting the list to a set allows us to verify whether a word is a stopword more quickly since looking up things in a set in Python is faster than looking up items in a list. This will be beneficial during the text preparation step, where we usually eliminate these stopwords.


### Remove common meaningless words of the text
Below figure is the content of MD&A test and risk factor text that we extracted from the 10K report, next we have to do text filtering and cleansing work.

![Picture showing Powell]({static}/images/NLP-Power_03_image-1.png)

This function, remove_first_sentence(text), is intended to tidy up the input text by deleting phrases and words that are considered unneeded or useless for the analysis. It matches and removes phrases and words using regular expressions (regex). The following function will be very helpful for preparing our 10-k text data, reducing unwanted noise and making the text clearer and simpler to analyse.

```python
# Function to remove first sentence and some common meaningless words of the text
def remove_first_sentence(text):

    subObj = re.sub( r'(.*our operations and financial results are subject to[\w\s\-\,\(\)\']+\.)', '', text, re.M|re.I)

    subObj = re.sub( r'(.*private securities litigation reform act of 1995[\w\s\-\,\(\)\']+\.)', '', text, re.M|re.I)

    subObj = re.sub( r'(.*forward-looking statement[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)

    subObj = re.sub( r'(.*managements discussion and analysis[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)

    subObj = re.sub( r'(.*the following discussion[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)

    subObj = re.sub( r'(.*this section[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)

    subObj = re.sub( r'\.(.*financial statements\s*\([\w,-]+\))', '', subObj, re.M|re.I)

    subObj = re.sub( r'\.(.*overview[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)

    subObj = re.sub( r'\.(.*statements other than[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)
  
    subObj = re.sub( r'\.(.*use words[\w\s\-\,\(\)\']+\.)', '', subObj, re.M|re.I)
 
    subObj = re.sub( r'part', '', subObj, re.M|re.I)

    subObj = re.sub( r'item', '', subObj, re.M|re.I)

    subObj = re.sub( r'i+', '', subObj, re.M|re.I)
    
    return subObj # paragraph
```

The function takes a string of text as input.

It then applies the re.sub() method from the re module to replace matching patterns with an empty string, essentially deleting them.

It matches patterns in certain sentences and words that we've determined aren't relevant to our investigation. These include basic sentences often found in financial reports and other formal documents, such as "our operations and financial results are subject to…", "private securities litigation reform act of 1995…", "forward-looking statement…", "managements discussion and analysis…", "the following discussion…", "this section…", "financial statements…", "overview…", "statements other than…", "use words…", "part", "item", and "i+".

The re.M|re.I flags in the re.sub() function make the matching process case-insensitive (re.I), and allow the ^ and $ metacharacters to work over several lines (re.M).

### Removal rare words and other common meaningless words
In order to Removal rare words we first need to count the frequency of occurrence of each word in the text, we will use the following function to help us to complete the statistics.
```python
def remove_rare_words(df):

    # Step 1: Calculate document frequency for each word
    word_counts = Counter()

    for row in df:
        try:
            words = set(word_tokenize(row))
            word_counts.update(words)
        except:
            continue
        
    # Step 2: Determine the total number of documents
    total_documents = len(df)

    # Step 3: Calculate frequency ratio for each word
    word_frequency_ratios = {word: count / total_documents for word, count in word_counts.items()}

    return word_frequency_ratios

```

Step 1: We sets a Counter object, word_counts, to keep track of the number of document in which each word appears. 

Step 2: It computes the total number of words in documents, which is the length of the DataFrame.

Step 3: The frequency ratio for each word is calculated as the number of documents in which the term appears divided by the total number of documents. 


Next we can remove rare words, Punctuation, Stopwords, 1- and 2- letter words, numbers, and finally we Lemmatise the remaining words.
```python
# Function for management discussion and analysis text preprocessing
def mda_preprocess_text(text):

    # Take away first sentence 
    text = remove_first_sentence(text)
    
    # Tokenization
    tokenized = word_tokenize(text)
    
    # Removal of rare words
    # tokens = [token for token in tokens if mda_word_freq_ratio[tokens] > 0.005]
    tokens = []
    for token in tokenized:
        try:
            if mda_word_freq_ratio[token] > 0.005:
                tokens.append(token)
        except KeyError:
            print(token)
            
    # Removal of Punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Removal of Stopword 
    tokens = [token for token in tokens if token not in stop_words]
    
    # Removal of 1- and 2- letter words
    tokens = [token for token in tokens if len(token) > 2]

    # Removal of numbers
    tokens = [token for token in tokens if not token.isnumeric()]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Joining the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    # Add Beginning & Ending tag
    
    print(len(preprocessed_text))
    return preprocessed_text
```

Through the above process, we have obtained the final cleaned text.