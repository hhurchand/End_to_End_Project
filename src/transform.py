class DataTransformation:
    def __init__(self,df, config):
        self.config = config
        self.df = df.copy()
        
    def check_input(self):
        print(self.config)