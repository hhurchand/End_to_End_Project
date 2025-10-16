import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.input import csvLoader, yamlLoader

import joblib
import nltk



class StreamlitController:

    

    """
    Initialize the StreamlitController object
    Load in the external parameters
    """
    def __init__(self):
        self.get_config()  
        
    
    
    def get_config(self):
        """
        Use the yamlLoader utility to fetch the external parameters stored in the params.yaml file
        """
        self.config = yamlLoader().load_file("params.yaml")

    
    def SetEmailContent(self, emailContent):
        """
        Take the user's email content and store it to a local variable
        Parameters
        -----------
        emailContent: The text content of the email to be classified
        """
        self.emailContent = emailContent

    
    def load_model(self):
        """
        Load in the most efficient model trained earlier that will be used in the email classification
        """
        model_location = self.config["data"]["pickle_file"]
        with open(model_location, 'rb') as file:
            self.model = pickle.load(file)

    
    def take_words_stem(self, text):
        """
        Take the user's text, remove all stop words, and break it down to just the stems of all alpha words included, 
        then store that to a local variable transformed_content
        Parameters
        -----------
        text: The unedited text found in the email
        """
        # Download NLTK stop words if not already downloaded
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')

        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        text = text.lower()

        wordsInText = text.split(" ")
        wordsInTextCleaned = []
        for j in range(len(wordsInText)):
            word = wordsInText[j]
            if word.isalpha() and word not in stop_words:
                wordsInTextCleaned.append(word)
        text = [stemmer.stem(word) for word in wordsInTextCleaned]
        text = " ".join(text)
        
        self.transformed_content = text

    
    def tokenize_text(self):
        """
        Take the transformed_content variable and tokenize it so it can be read by an ML algorithm
        """
        print("Tokenize")
        vectorizer_filepath = self.config["data"]["vectorizer"]
        self.vectorizer = joblib.load(vectorizer_filepath)
        # Convert the text to a bag-of-words representation
        corpus = [self.transformed_content]
        self.transformed_content = self.vectorizer.transform(corpus)


    
    def transform_email_content(self):
        """
        Stem the content of the email and tokenize it to be read by the ML algorithm
        """
        self.transformed_content = str(self.emailContent)
        self.take_words_stem(self.transformed_content)
        self.tokenize_text()


    
    def predict_email(self):
       """
        Classify the cleaned-up email as either Spam or Ham using the ML model
        """
       self.y_pred = self.model.predict(self.transformed_content)
       print(self.y_pred)
       return self.y_pred
