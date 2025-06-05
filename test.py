import argparse
import numpy as np
from model import NeuralNetwork
from utils import DatasetLoader, Config
from sklearn.metrics import precision_score, recall_score, f1_score

class Tester:
    def __init__(self, config):
        self.model = NeuralNetwork(config.input_size, config.hidden_size, config.output_size)
        weights = NeuralNetwork.load_weights(config.weights_path)
        self.model.dense1.weights, self.model.dense1.bias = weights[0], weights[1]
        self.model.dense2.weights, self.model.dense2.bias = weights[2], weights[3]

    def test(self, X, Y):
        predictions = self.model.predict(X)
        true_classes = np.argmax(Y, axis=1)

        accuracy = np.mean(predictions == true_classes)
        precision = precision_score(true_classes, predictions, average='macro', zero_division=0)
        recall = recall_score(true_classes, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_classes, predictions, average='macro', zero_division=0)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

def main():
    config = Config.load_config()

    parser = argparse.ArgumentParser(description="Тестирование нейронной сети.")
    parser.add_argument("--data_path", type=str, required=True, help="Путь до данных.")
    args = parser.parse_args()

    X_test, y_test = DatasetLoader.load_images_from_folder(args.data_path, dataset_type='test')
    y_test_onehot = DatasetLoader.one_hot_encode(y_test, config.output_size)

    tester = Tester(config)
    tester.test(X_test, y_test_onehot)

if __name__ == "__main__":
    main()
