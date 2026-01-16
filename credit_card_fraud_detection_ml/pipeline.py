import pickle
import pandas as pd

class FraudPredictionPipeline:

    def __init__(self):
        self.scaler = None
        self.model = None

    def load_artifacts(self):
        with open("artifacts/scaler.pickle", "rb") as file:
            self.scaler = pickle.load(file)

        with open("artifacts/model.pickle", "rb") as file:
            self.model = pickle.load(file)

    def preprocess(self, data: pd.DataFrame):

        columns_to_use = self.scaler.feature_names_in_
        data_to_scale = data[columns_to_use]
        return self.scaler.transform(data_to_scale)

    def predict(self, data: pd.DataFrame):
        self.load_artifacts()
        processed_data = self.preprocess(data)
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)[:, 1]
        return prediction, probability

input_data = {
    "Time": [57027],
    "Amount": [444.17],
    "V1": [-2.335654929],
    "V2": [2.225379941],
    "V3": [-3.379450386],
    "V4": [2.178538229],
    "V5": [-3.568263715],
    "V6": [0.316813584],
    "V7": [-1.734948003],
    "V8": [1.449139434],
    "V9": [-1.980033481],
    "V10": [-5.71150474],
    "V11": [1.837214902],
    "V12": [-4.540341645],
    "V13": [0.74784586],
    "V14": [-6.28431407],
    "V15": [-0.128887217],
    "V16": [-3.563239406],
    "V17": [-7.368320646],
    "V18": [-2.692953158],
    "V19": [-0.450550114],
    "V20": [0.274027123],
    "V21": [0.785540122],
    "V22": [0.297411984],
    "V23": [0.308536146],
    "V24": [-0.598415746],
    "V25": [-0.121850331],
    "V26": [-0.49101831],
    "V27": [0.701606041],
    "V28": [0.206966343]
}

df = pd.DataFrame(input_data)
pipeline = FraudPredictionPipeline()
pred, prob = pipeline.predict(df)

print("Prediction:", "Fraud" if pred[0] == 1 else "Not Fraud")
print("Fraud Probability:", prob[0])