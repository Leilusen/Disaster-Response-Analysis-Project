from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
wpt = nltk.WordPunctTokenizer()
import pandas as pd
import re

def tokenize(text):
    '''
    Function for tokenizing text data
    Parameters:
    Text message
    Returns:
    Tokenized data
    
    '''
    stop_words = nltk.corpus.stopwords.words('english')

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    tokens = wpt.tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

class CustomTransformer(BaseEstimator, TransformerMixin):
    '''
    Class for transforming tokenized data into dataframe of new features 
    that can be fed into the model
    Parameters:
    BaseEstimator and TransformerMixin classes
    Returns:
    Pandas dataframe containing new features
    
    '''

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # index pos_tags to get the first word and part of speech tag
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
            # return true if the first word is an appropriate verb or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1.
            return 0.
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tokenized = pd.Series(X).apply(lambda x: tokenize(x))
        X_exclam = pd.Series(X).apply(lambda x: x.count('!'))
        X_question = pd.Series(X).apply(lambda x: x.count('?'))
        X_unique = X_tokenized.apply(lambda x: len(set(w for w in x)))
        X_words = X_tokenized.apply(lambda x : len(x))
        X_unique_freq = X_unique / X_words
        X_verbs = X_tokenized.apply(lambda x: len([token[0] for token in 
                                                    nltk.pos_tag(x) if token[1] in ['VB', 'VBP']]))
        X_verbs_freq = X_verbs / X_words
        
        X_nouns = X_tokenized.apply(lambda x: len([token[0] for token in nltk.pos_tag(x) if token[1] == 'NN']))
        X_nouns_freq = X_nouns / X_words

        return pd.concat([X_tagged, X_exclam, X_question, X_unique_freq, X_verbs_freq, X_nouns_freq], axis = 1).fillna(0)
