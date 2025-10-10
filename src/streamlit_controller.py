import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.input import csvLoader, yamlLoader

import joblib

class StreamlitController:

    def __init__(self):
          self.get_config()  
    
    def get_config(self):
        self.config = yamlLoader().load_file("params.yaml")

    def SetEmailContent(self, emailContent):
        self.emailContent = emailContent

    def load_model(self):
        model_location = self.config["data"]["pickle_file"]
        with open(model_location, 'rb') as file:
            self.model = pickle.load(file)

    def take_words_stem(self, text):
        stemmer = PorterStemmer()
        text = text.lower()

        wordsInText = text.split(" ")
        wordsInTextCleaned = []
        for j in range(len(wordsInText)):
            word = wordsInText[j]
            if word.isalpha():
                wordsInTextCleaned.append(word)
        text = [stemmer.stem(word) for word in wordsInTextCleaned]
        text = " ".join(text)
        
        self.transformed_content = text

    def tokenize_text(self):
        print("Tokenize")
        vectorizer_filepath = self.config["data"]["vectorizer"]
        self.vectorizer = joblib.load(vectorizer_filepath)
        # Convert the text to a bag-of-words representation
        corpus = [self.transformed_content]
        self.transformed_content = self.vectorizer.transform(corpus)


    def transform_email_content(self):
        self.transformed_content = str(self.emailContent)
        self.take_words_stem(self.transformed_content)
        self.tokenize_text()


    def predict_email(self):
       self.y_pred = self.model.predict(self.transformed_content)
       print(self.y_pred)
       return self.y_pred
