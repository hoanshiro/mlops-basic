import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from data import DataModule
from utils import timing

class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0)
        }
        ort_output = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_output[0])[0]
        prediction = []
        for score, label in zip(scores, self.labels):
            prediction.append({"label": label, "score": score})
        return prediction
    
if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./outputs/models/model.onnx")
    print(predictor.predict(sentence))
    sentences = [sentence]*10
    for sentence in sentences:
        predictor.predict(sentence)
