import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.input import csvLoader, yamlLoader

import joblib

class StreamlitController:

    """
    Initialize the StreamlitController object
    Load in the external parameters
    """
    def __init__(self):
          self.get_config()  
    
    """
    Use the yamlLoader utility to fetch the external parameters stored in the params.yaml file
    """
    def get_config(self):
        self.config = yamlLoader().load_file("params.yaml")

    """
    Take the user's email content and store it to a local variable
    Parameters
    -----------
    emailContent: The text content of the email to be classified
    """
    def SetEmailContent(self, emailContent):
        self.emailContent = emailContent

    """
    Load in the most efficient model trained earlier that will be used in the email classification
    """
    def load_model(self):
        model_location = self.config["data"]["pickle_file"]
        with open(model_location, 'rb') as file:
            self.model = pickle.load(file)

    """
    Take the user's text and break it down to just the stems of all alpha words included, 
    then store that to a local variable transformed_content
    Parameters
    -----------
    text: The unedited text found in the email
    """
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

    """
    Take the transformed_content variable and tokenize it so it can be read by an ML algorithm
    """
    def tokenize_text(self):
        print("Tokenize")
        vectorizer_filepath = self.config["data"]["vectorizer"]
        self.vectorizer = joblib.load(vectorizer_filepath)
        # Convert the text to a bag-of-words representation
        corpus = [self.transformed_content]
        self.transformed_content = self.vectorizer.transform(corpus)


    """
    Stem the content of the email and tokenize it to be read by the ML algorithm
    """
    def transform_email_content(self):
        self.transformed_content = str(self.emailContent)
        self.take_words_stem(self.transformed_content)
        self.tokenize_text()


    """
    Classify the cleaned-up email as either Spam or Ham using the ML model
    """
    def predict_email(self):
       self.y_pred = self.model.predict(self.transformed_content)
       print(self.y_pred)
       return self.y_pred
