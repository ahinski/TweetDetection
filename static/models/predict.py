import nltk
import re
import pickle
import os
import math
import string
import numpy as np

class App():
    ''' Class for API application 'Probability of Disaster Tweet'

    Attributes:
    tweet - text from tweet
    
    Methods:
    calculate() returns probability in json form {probability: result}
    '''
    def __init__(self, tweet):
        self.text = Text(tweet)

    def calculate(self):
        'Calculates the probability of tweet being real about disaster and return JSON'
        clf_ensamble = pickle.load(open('static/models/ens_model.sav', 'rb'))
        text = self.text.preprocess()
        result = math.floor(clf_ensamble.predict_proba(text)[0, 1] * 100) / 100
        return result

class Text():
    ''' Class for representing text

    Attributes:
    text - tweet text
    
    Methods:
    preprocess - for preprocessing provided text to fit in ML model in class App
    '''
    def __init__(self, text):
        self.text = text
        self.__preprocessed = False

    #Lemmatizes text
    def __lemmatize(self):
        'Lemmatizes text'
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(self.text)])

    def preprocess(self):
        'Preprocesses text for fitting in ML model'
        if (self.__preprocessed == False):
            features = self.__meta_features()

            # lowercase
            self.text = self.text.lower()

            # removing links, mentions, and hashtag (only the symbol)
            self.text = re.sub(r'https?://\S+|www\.\S+','', self.text)
            self.text = re.sub(r'@[A-Za-z0-9]+','', self.text)
            self.text = re.sub(r'#','', self.text)

            # removing punctuation
            self.text = re.sub(r'[^\w\s]','', self.text)

            # lemmatizing text
            self.text = self.__lemmatize()

            # vectorizing text
            vect = pickle.load(open('static/models/vectorizer.sav', 'rb'))
            self.text = vect.transform([self.text])
            self.text = self.text.todense()

            # concatenating meta_features and word-vector
            self.text = np.concatenate((self.text, features), axis=1)
            self.__preprocessed = True
        return self.text

    def __meta_features(self):
        'Extracts meta features from text'
        #Mentions count
        if self.text.count('@'): mentions_count = 1
        else: mentions_count = 0

        #Links count
        if self.text.count('http'): links_count = 1
        else: links_count = 0

        #Punctuation count
        count = lambda l1,l2: sum([1 for x in l1 if x in l2])
        punct_count = count(self.text, string.punctuation)

        features = np.array([mentions_count, links_count, punct_count])
        return features.reshape(1, 3)