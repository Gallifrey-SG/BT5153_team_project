'''data processing script''' 

import re
import string
import numpy as np
import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from ftfy import fix_text
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer
from models.data_utils import TextDataset

# NLTK downloads for text processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TextProcessor:
    """ processes text data and extract features """

    def __init__(self):
        """Initialize the processor with necessary setups, including loading stopwords."""
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def normalize_text(self, text):
        """Normalize the text by fixing text encoding, converting to lowercase, removing punctuation, and collapsing whitespace."""
        text = fix_text(text)  # Fix text encoding issues
        text = text.lower()  # Convert text to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\s{2,}', ' ', text)  # Collapse multiple spaces
        return text

    def remove_stopwords(self, text):
        """Remove stopwords from the text."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_text_to_df(self, text):
        normalized_text = self.normalize_text(text)
        # Assuming the DataFrame structure matches your model's expected input
        data = {
            'text': [normalized_text],
            'label': [0]  # Dummy label if needed
        }
        df = pd.DataFrame(data)
        return df
    
    def prepare_data_loader(self, text):
        df = self.preprocess_text_to_df(text)
        dataset = TextDataset(df, self.tokenizer)
        loader = DataLoader(dataset, batch_size=1)
        return loader

    def extract_features(self, text):
        """Extract and return specified text features for analysis and predictions as a DataFrame."""
        normalized_text = self.normalize_text(text)
        clean_text = self.remove_stopwords(normalized_text)
        sentence_variation = self.sentence_length_variation(clean_text)
        adjective_count = self.count_adjectives(clean_text)
        word_count = len(clean_text.split())
        stop_word_count = len([w for w in clean_text.split() if w in self.stop_words])
        percent_stop_words = stop_word_count / word_count if word_count else 0
        mean_word_length = np.mean([len(word) for word in clean_text.split()]) if clean_text.split() else 0
        percent_adjectives = adjective_count / word_count if word_count else 0

        # Create a DataFrame with the extracted features
        data = {
            'text': [clean_text],
            'label': [0],  # Placeholder, set as needed
            'sentence_variation': [sentence_variation],
            '%stop_word_total': [percent_stop_words],
            'mean_word_length': [mean_word_length],
            '%adjectives_total': [percent_adjectives]
        }

        return pd.DataFrame(data)

    def sentence_length_variation(self, text):
        """Calculate the standard deviation of sentence lengths in the text."""
        sentences = sent_tokenize(text)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        return np.std(sentence_lengths)

    def count_adjectives(self, text):
        """Count the number of adjectives in the text."""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return len([word for word, pos in pos_tags if pos.startswith('JJ')])