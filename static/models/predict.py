import nltk
import re
import pickle
import os
import math
from flask import jsonify


class App():
    ''' Class for API application 'Probability of Disaster Tweet'

    Attributes:
    tweet - tweet in json form {tweet: str(text)}
    
    Methods:
    calculate() returns probability in json form {probability: result}
    '''
    def __init__(self, tweet):
        self.text = Text(tweet.tweet)

    def calculate(self):
        'Calculates the probability of tweet being real about disaster and return JSON'
        clf_LogReg = pickle.load(open('LogReg_model.sav', 'rb'))
        clf_RF = pickle.load(open('RF_model.sav', 'rb'))
        clf_NB = pickle.load(open('NB_model.sav', 'rb'))
        text = self.text.preprocess()
        predict_LogReg = clf_LogReg.predict_proba(text)[:, 1]
        predict_RF= clf_RF.predict_proba(text)[:, 1]
        predict_NB = clf_NB.predict_proba(text)[:, 1]
        result = math.floor(((predict_LogReg + predict_RF + predict_NB) / 3)[0] * 100) / 100
        return jsonify(
                    {"probability": str(result)}
                ),

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
            # removing stopwords
            all_stopwords = set(nltk.corpus.stopwords.words('english'))
            self.text = ' '.join([word for word in self.text.split() if word not in (all_stopwords)])
            # removing words with numbers and letters along
            self.text = ' '.join(w for w in self.text.split() if not any(j.isdigit() for j in w))
            vect = pickle.load(open('vectorizer.sav', 'rb'))
            self.text = vect.transform([self.text])
            self.text = self.text.todense()
            self.__preprocessed = True
        return self.text