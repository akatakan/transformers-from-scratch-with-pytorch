import torch
import numpy as np
import pandas as pd
import re

class ScratchTokenizer():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[w] for w in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
        

text = """In the previous section, we tokenized Edith Wharton's short story and
assigned it to a Python variable called preprocessed. Let's now create a list
of all unique tokens and sort them alphabetically to determine the
vocabulary size """

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if
item.strip()]

words = sorted(set(preprocessed))
vocab = {token:integer for integer,token in enumerate(words)}

tokenizer = ScratchTokenizer(vocab)

print(tokenizer.encode("Python variable called preprocessed"))
print(tokenizer.decode([6, 34, 13, 20]))