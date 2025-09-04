from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Model:
    def __init__(self,df,config):
        self.df = df.copy()
        self.config = config

    def split_data(self,X,y,test_size=0.3,random_state=42):
        self.X=X
        self.y=y
        self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=test_size,random_state=random_state)
        return self.X_train,self.X_test,self.y_train,self.y_test

    def modeling(self,random_state=42):
        self.split_data(self.X,self.y)
        model = RandomForestRegressor(random_state=random_state)
        self.model = model.fit(self.X_train,self.y_train)
        return self.model
    
    def prediction(self):
        y_pred = self.model.predict(self.X_test)
        print("Mean squared error (MSE) =", metrics.mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error (MAE) =", metrics.mean_absolute_error(self.y_test, y_pred))
        print("Root Mean Square =", metrics.root_mean_squared_error(self.y_test, y_pred))