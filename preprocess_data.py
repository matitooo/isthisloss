import string 
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def to_lowercase(text):
    """ Convert text to lowercase """
    return text.lower()

def remove_punctuation(text):
    """ Remove punctuation from text """
    text = re.sub(r"[^\w\s]", "", text)
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    """ Remove stopwords from text """
    stop_words = set(stopwords.words('english'))
    words=text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """ Lemmatize text """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_hashtag_words(text):
    """ Remove hashtags """
    return re.sub(r'\s*#\S+', '', text).strip()

def preprocess_text(text):
    """ Preprocess text: lowercase, remove punctuation, remove stopwords, lemmatize"""
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = remove_hashtag_words(text)
    return text


