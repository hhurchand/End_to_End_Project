import pandas as pd

class Mod_dataset:
    """
    A class to split a single-column DataFrame containing 'label,message' strings
    into separate 'label' and 'Message' columns.

    """

    def __init__(self,df):
        """
        Initialize the Mod_dataset with a DataFrame
        """
        self.data = df.copy()


    def process(self):
        """
        Split the single column DataFrame into "Label" and Message" columns
                
        Returns:
            pd.DataFrame: Modified DataFrame with 'label' and 'Message' columns.
        """
        # Rename the only column to 'text' 
        self.data.columns = ['text']
        
        # Split the 'text' column into 'label' and 'Message' on the first comma
        self.data[['label', 'Message']] = self.data['text'].str.split(',', n=1, expand=True)
        
        # Drop the original 'text' column
        self.data.drop(columns=['text'], inplace=True)
        
        return self.data