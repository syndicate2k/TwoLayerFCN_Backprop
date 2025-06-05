import argparse
import numpy as np
from model import NeuralNetwork
from utils import DatasetLoader, Config

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = NeuralNetwork(config.input_size, config.hidden_size, config.output_size)

    def train(self, X, Y):
        losses = []
        for epoch in range(self.config.epochs):
            output = self.model.forward(X)
            loss = self.model.compute_loss(Y, output)
            losses.append(loss)
            self.model.backward(X, Y, self.config.learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

        return losses

def main():
    config = Config()

    parser = argparse.ArgumentParser(description="Обучение нейронной сети.")
    parser.add_argument("--data_path", type=str, required=True, help="Путь до данных.")
    parser.add_argument("--input_size", type=int, default=config.input_size, help="Размер входного слоя.")
    parser.add_argument("--hidden_size", type=int, default=config.hidden_size, help="Кол-во нейронов в скрытом слое.")
    parser.add_argument("--output_size", type=int, default=config.output_size, help="Размер выходного слоя.")
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Кол-во эпох.")
    parser.add_argument("--weights_path", type=str, default=config.weights_path, help="Путь для сохрания весов.")

    args = parser.parse_args()

    config.input_size = args.input_size
    config.hidden_size = args.hidden_size
    config.output_size = args.output_size
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
    config.weights_path = args.weights_path

    X_train, y_train = DatasetLoader.load_images_from_folder(args.data_path, dataset_type='train')
    y_train_onehot = DatasetLoader.one_hot_encode(y_train, config.output_size)

    trainer = Trainer(config)
    losses = trainer.train(X_train, y_train_onehot)

    trainer.model.save_weights(config.weights_path)
    config.save_config()
