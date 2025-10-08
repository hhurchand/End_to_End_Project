from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas as pd

from scipy import sparse
import joblib
class DataTransformation:
    def __init__(self,df, config):
        self.config = config
        self.df = df.copy()
        

    def remove_na(self):
        print("Removing NA")
        self.df = self.df.dropna()

    def rename_spam_column(self):
        print("Removing spam column")
        self.df['Spam'] = self.df['Spam/Ham']
        self.df.drop('Spam/Ham',  axis =1, inplace=True)

    def make_spam_column_boolean(self):
        print("Making spam column boolean")
        self.df["Spam_Bool"] = (self.df["Spam"] =="spam").astype(int)

    def declare_x_y_fields(self):
        print("Declare X/Y fields")
        self.X = self.df[["Message"]].copy()
        self.y = self.df["Spam_Bool"].copy()

    def take_words_stem(self):
        print("Take Words Stem")
        stemmer = PorterStemmer()

        wordsLibrary = []
        for i in range(len(self.X)):
            text = self.X["Message"].iloc[i].replace("\n"," ")

            text = text.lower()

            wordsInText = text.split(" ")
            wordsInTextCleaned = []
            for j in range(len(wordsInText)):
                word = wordsInText[j]
                if word.isalpha():
                    wordsInTextCleaned.append(word)
            text = [stemmer.stem(word) for word in wordsInTextCleaned]
            text = " ".join(text)
            wordsLibrary.append(text)


        self.X = wordsLibrary
        print(len(self.X))

    
    def tokenize_text(self):
        print("Tokenize")
        # Convert the text to a bag-of-words representation
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)

    def save_data(self):
        
        print("Save data to csv")
        
        #with open('data/cleaned/output.csv', 'w', newline='') as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow(['Message', 'Spam_Bool'])  # Write header row
        #    dense_matrix = self.X.toarray()
        #    print(f"X length: {len(dense_matrix)}")
        #    print(f"y length: {len(self.y)}")
        #    for i in range(len(dense_matrix)):
        #        writer.writerow([dense_matrix[i], self.y.iloc[i]])

        vectorizer_path = self.config["data"]["vectorizer"]
        X_sparse_path = self.config["data"]["X_sparse"]
        cleaned_y_path = self.config["data"]["cleaned_y"]
        
        joblib.dump(self.vectorizer, vectorizer_path)
        sparse.save_npz(X_sparse_path, self.X)

        self.y.to_csv(cleaned_y_path, index=False)

        print("done saving")
                



    def transform_data_pipeline(self):
        self.remove_na()
        self.rename_spam_column()
        self.make_spam_column_boolean()

        self.declare_x_y_fields()

        self.take_words_stem()

        self.tokenize_text()

        self.save_data()

        return self.X, self.y
