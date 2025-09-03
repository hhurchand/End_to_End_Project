from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class ModelTraining:
    def __init__(self,df,config):
        self.df = df.copy()
        self.config = config

    def split_data(self,X,y):
        X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        return X_train,y_train,X_test,y_test

    def modeling(self,model):
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train,self.y_train)
        return model