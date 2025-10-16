from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle

class Train_Split_data:
    def __init__(self, features, target, test_size=0.2, val_size=0.1):
        self.features = features
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
    
    def train_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, 
            self.target, 
            test_size=self.test_size, 
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.val_size,
            random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

class Modeling:
    @staticmethod
    def functional_api(input_shape,dense1_no,dense2_no,dropout,activation,output_activation):
        input_layer = Input(shape=input_shape)
        dense1 = Dense(dense1_no, activation=activation)(input_layer)
        dropout1 = Dropout(dropout)(dense1)
        batch_norm1 = BatchNormalization()(dropout1)
        dense2 = Dense(dense2_no, activation=activation)(batch_norm1)
        output_layer = Dense(1, activation=output_activation)(dense2)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    
    @staticmethod
    def compiling(model, X_train, y_train, X_val, y_val, epochs, batch_size):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )
        return model, history

    @staticmethod
    def evaluate(model, X_test, y_test, threshold=0.5):
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\nTest Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        return acc, report, y_pred
    
    def save_model(self, model, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")

class ML_Split_data:
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def xy_train(self):
        return train_test_split(self.features,self.target, test_size=0.3, random_state=42) 

class ML_Modeling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_selection(self,model):
        model.fit(self.X_train,self.y_train)
        y_pred = model.predict(self.X_test)
        print(f"The accuracy score of {model} : {accuracy_score(self.y_test, y_pred)}")
        print(classification_report(self.y_test, y_pred))

class ML_Evaluation(Modeling):
    def __init__(self, X_train, X_test, y_train, y_test, select_model):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.select_model = select_model

    def model_validation(self):
        validate = cross_val_score(self.select_model, self.X_test, self.y_test, cv=5)
        print(f"Validation : {self.select_model}{validate}")
