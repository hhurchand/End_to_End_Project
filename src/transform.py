from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas as pd

from scipy import sparse
import joblib

from nltk.corpus import stopwords
import nltk
class DataTransformation:
    """
    Initializes DataTransformation Object

    Parameters
    -----------
    df: The dataframe featuring the data to be transformed
    config: A dictionary of modifiable variables to be referenced
    """
    def __init__(self,df, config):
        self.config = config
        self.df = df.copy()
        
    """
    Removes any NA values from the dataframe
    """
    def remove_na(self):
        print("Removing NA")
        self.df = self.df.dropna()

    """
    Removes the Spam/Ham column and replaces it with one just named Spam
    """
    def rename_spam_column(self):
        print("Removing spam column")
        self.df['Spam'] = self.df['Spam/Ham']
        self.df.drop('Spam/Ham',  axis =1, inplace=True)

    """
    Makes a new column in our dataframe, Spam_Bool that remaps Spam or Ham values to 1 or 0
    """
    def make_spam_column_boolean(self):
        print("Making spam column boolean")
        self.df["Spam_Bool"] = (self.df["Spam"] =="spam").astype(int)

    """
    Declare the Message field as the X value and the Spam_Bool field as our y value
    """
    def declare_x_y_fields(self):
        print("Declare X/Y fields")
        self.X = self.df[["Message"]].copy()
        self.y = self.df["Spam_Bool"].copy()

    """
    Iterate through the emails/messages and keep only Alpha words (removing any puctuation or numeric values)
    Then, replace the words with just their stem for easier reading by the algorithm
    """
    def take_words_stem(self):
        print("Take Words Stem")
        stemmer = PorterStemmer()

        # Get English stop words
        stop_words = set(stopwords.words('english'))

        wordsLibrary = []
        for i in range(len(self.X)):
            text = self.X["Message"].iloc[i].replace("\n"," ")

            text = text.lower()

            wordsInText = text.split(" ")
            wordsInTextCleaned = []
            for j in range(len(wordsInText)):
                word = wordsInText[j]
                if word.isalpha() and word not in stop_words:
                    wordsInTextCleaned.append(word)
            text = [stemmer.stem(word) for word in wordsInTextCleaned]
            text = " ".join(text)
            wordsLibrary.append(text)


        self.X = wordsLibrary
        print(len(self.X))

    """
    Take the stemmed words and tokenize them so that they can be read by a machine learning model
    """
    def tokenize_text(self):
        print("Tokenize")
        # Convert the text to a bag-of-words representation
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)


        """
        Save the word vectorizer trained earlier, the tokenized words, and the Spam/Ham boolean values to different files
        """
    def save_data(self):
        
        print("Save data to csv")
        vectorizer_path = self.config["data"]["vectorizer"]
        X_sparse_path = self.config["data"]["X_sparse"]
        cleaned_y_path = self.config["data"]["cleaned_y"]
        
        joblib.dump(self.vectorizer, vectorizer_path)
        sparse.save_npz(X_sparse_path, self.X)

        self.y.to_csv(cleaned_y_path, index=False)

        print("done saving")
                


        """
        Perform the entire data transformation pipeline
        Remove NA
        Rename Spam/Ham Column
        Remap the Spam column to a boolean equivalent

        Declare our X and y fields that will be transformed for the model training
        Stem the words in our X field
        Tokenize the words in our X field

        Save the transformed data for the next run
        """
    def transform_data_pipeline(self):
        self.remove_na()
        self.rename_spam_column()
        self.make_spam_column_boolean()

        self.declare_x_y_fields()

        self.take_words_stem()

        self.tokenize_text()

        self.save_data()

        return self.X, self.y
