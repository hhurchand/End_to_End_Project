import pandas as pd
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class Transform:
    def __init__(self):
        pass
    
    def dropnull(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data
        data = data.dropna(subset=["text"])
        return data

    def transform_text(self,data: pd.DataFrame):
        self.data = data
        # lowercase words in text column
        self.data["text"] = self.data["text"].apply(lambda x:x.lower())
        # Get rid of all numbers
        self.data["text"] = self.data["text"].apply(lambda x:re.sub(r"\d+","",x))
        # Remove whitspaces
        self.data["text"] = self.data["text"].apply(lambda x : x.strip())

        for punc in string.punctuation:
            self.data["text"] = self.data["text"].apply(lambda x: x.replace(punc, ""))

        return self.data
        
class Tokenized_sentence:
    def __init__(self):
        pass
    @staticmethod
    def tokenize_text(text:str):
        tokens = word_tokenize(text)
        english_stop_words = stopwords.words("english")
        tokens = [word for word in tokens if word not in english_stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens
    

class Label_column:
  def __init__(self,data:pd.DataFrame):
    self.data = data

  def label(self):
    label_enc = LabelEncoder()
    self.data["label"] = label_enc.fit_transform(self.data["label"])
    return self.data
  

class Tfidf(Label_column):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.data = data

    def tfidvec(self):
        self.data["text"] = self.data["text"].apply(lambda x: " ".join(x))

        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform(self.data["text"])
        y = self.data["label"]
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        return X, y
    
