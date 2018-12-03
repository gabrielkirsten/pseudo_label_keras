import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from metrics.Metric import Metric


class LearningCurve(Metric):
    def __init__(self):
        pass

    @staticmethod
    def obtain(data_points_to_learning_curve, plot_graphic=False):
        data_points_qtd_examples = []
        data_points_accuracy = []
        for data_point in data_points_to_learning_curve:
            accuracy = accuracy_score(data_point.get('output_real'), data_point.get('output_predict'))
            data_points_qtd_examples.append(data_point.get('qtd_examples'))
            data_points_accuracy.append(accuracy)

            if not plot_graphic:
                print("Qtd examples %d - Accuracy: %f" % (data_point.get('qtd_examples'), accuracy))
        
        if plot_graphic:
            plt.figure(1)
            plt.plot(data_points_qtd_examples, data_points_accuracy, 'ob-')
            plt.show()
            
        
