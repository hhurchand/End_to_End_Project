from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class Model:

    def __init__(self,df_encoded, config):
        self.df_encoded = df_encoded.copy()
        self.config = config


    def train_test_split(self):
        final_features = list(self.df_encoded.columns)

        targetfeature = self.config["model"]["target"]
        final_features.remove(targetfeature)
        X = self.df_encoded[final_features]
        y= self.df_encoded[targetfeature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.3, random_state=42)

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(self.X_train,self.y_train)
        return self.model

    def PredictPrice(self):
        predicted_price = self.model.predict(self.X_test)
        print(f"Predicted price: {predicted_price}")
