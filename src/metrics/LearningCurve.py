from metrics.Metric import Metric
from sklearn.metrics import accuracy_score


class LearningCurve(Metric):
    def __init__(self):
        pass

    @staticmethod
    def obtain(self, output_predict, output_real):
        accuracy = accuracy_score(output_real, output_predict)
        
