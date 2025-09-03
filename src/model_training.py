from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


class ModelTraining:
    def __init__(self,df,config):
        self.df = df.copy()
        self.config = config

    def split_data(self,X,y):
        X = self.df_encoded.drop(config=["model"]["target"],axis=1)
        y = self.df_encoded(config=["model"]["target"])
        X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        return X_train,y_train,X_test,y_test

    def modeling(self,model):
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train,self.y_train)
        return model
    
    def prediction(self):
        y_pred = self.model.predict(self.X_test)
        print("Mean squared error (MSE) =", metrics.mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error (MAE) =", metrics.mean_absolute_error(self.y_test, y_pred))
        print("Root Mean Square =", metrics.root_mean_squared_error(self.y_test, y_pred))