from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class ModelTraining:

    def __init__(self,df_encoded, config):
        self.df_encoded = df_encoded.copy()
        self.config = config


    def TrainModel(self):
        final_features = list(self.df_encoded.columns)

        targetfeature = self.config["model"]["target"]
        final_features.remove(targetfeature)
        X = self.df_encoded[final_features]
        y= self.df_encoded[targetfeature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.3, random_state=42)
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(self.X_train,self.y_train)

    def PredictPrice(self):
        predicted_price = self.model.predict(self.X_test)
