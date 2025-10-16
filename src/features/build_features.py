import pandas as pd
import string
from typing import List
import string
from yaml import safe_load
from constants import PARAMS
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Required nltk ressources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("stopwords")


with open(PARAMS,encoding="UTF-8") as f:
    """
    Load configuration parameters from a YAML file specified by PARAMS.

    The configuration is loaded once and stored in `config`.
    """
    config = safe_load(f)



class DataCleaning():
    """
    Class for performing basic data cleaning operations on pandas DataFrames.

    Attributes:
        data (pd.DataFrame): The input DataFrame to be cleaned.
    """

    def __init__(self,frame: pd.DataFrame):
        """
        Initialize DataCleaning with a DataFrame.

        Args:
            frame (pd.DataFrame): DataFrame containing the data to clean.
        """
        self.data=frame


    def lower_case(self,columns:List[str]) -> pd.DataFrame:
        """
        Convert all text in specified columns to lowercase.

        Args:
            columns (List[str]): List of column names to convert.

        Returns:
            pd.DataFrame: DataFrame with specified columns converted to lowercase.
        """
        for column in columns:
            self.data[column] = self.data[column].str.lower()
        return self.data
    

    def remove_punc(self,columns:List[str]) -> pd.DataFrame:
        """
        Remove punctuation characters from specified columns.

        Args:
            columns (List[str]): List of column names to process.

        Returns:
            pd.DataFrame: DataFrame with punctuation removed from specified columns.
        """
        punc = list(string.punctuation)
        for words in columns:
            self.data[words] = self.data[words].apply(lambda x: "".join(char for char in x if char not in punc))
        return self.data
    

    def remove_space(self,columns:List[str]) -> pd.DataFrame:
        """
        Remove extra spaces from specified columns.

        Args:
            columns (List[str]): List of column names to process.

        Returns:
            pd.DataFrame: DataFrame with spaces normalized in specified columns.
        """
        for sent in columns:
            self.data[sent] = self.data[sent].apply(lambda x: " ".join(x.split()))
        return self.data
    
    
class Features():
    """
    Class for feature transformations on text data.

    Attributes:
        data (pd.DataFrame): The DataFrame containing features to transform.
    """

    def __init__(self,frame:pd.DataFrame):
        """
        Initialize Features with a DataFrame.

        Args:
            frame (pd.DataFrame): DataFrame containing the features.
        """
        self.data = frame


    def text_normalization(self,columns:list[str]) -> pd.DataFrame:
        """
        Normalize text data by tokenizing, removing stopwords, lemmatizing,
        and removing single-character tokens in specified columns.

        Args:
            columns (List[str]): List of column names to normalize.

        Returns:
            pd.DataFrame: DataFrame with normalized text in specified columns.
        """
        english_stop_words = stopwords.words("english")
        lemmatizer = WordNetLemmatizer()

        for col in columns:
            self.data[col] = self.data[col].apply(lambda x: [word for word in word_tokenize(str(x)) if word not in english_stop_words and len(word) > 1])

        for col in columns:
            self.data[col] = self.data[col].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x if len(lemmatizer.lemmatize(word)) > 1))

        return self.data
    

    def encoding(self,columns:List[str]) -> pd.DataFrame:
        """
        Encode categorical labels in specified columns using label encoding.

        Args:
            columns (List[str]): List of column names to encode.

        Returns:
            pd.DataFrame: DataFrame with encoded label columns.
        """
        for col in columns:
            label_enc = LabelEncoder()
            self.data[col]= label_enc.fit_transform(self.data[col])
        return self.data
    

    def transform_text_num(self,columns:List[str]) -> tuple:
        """
        Apply TF-IDF vectorization on specified text columns.

        Args:
            columns (List[str]): List containing one column name to transform.

        Returns: Tuple
            scipy.sparse.csr_matrix: TF-IDF feature matrix and
            fitted TfidfVectorizer object
        """
        max_feature = config["transform"]["max_feature"]
        tfidf_vectorize = TfidfVectorizer(max_features=max_feature)
        text_data = self.data[columns[0]]
        X = tfidf_vectorize.fit_transform(text_data)
        return X, tfidf_vectorize
