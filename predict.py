import argparse
import cv2
import numpy as np
from model import NeuralNetwork
from utils import Config

class Predictor:
    def __init__(self, config):
        self.model = NeuralNetwork(config.input_size, config.hidden_size, config.output_size)
        weights = NeuralNetwork.load_weights(config.weights_path)
        self.model.dense1.weights, self.model.dense1.bias = weights[0], weights[1]
        self.model.dense2.weights, self.model.dense2.bias = weights[2], weights[3]

    def predict(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.flatten().astype('float32') / 255.0
        img = img.reshape(1, -1)
        prediction = self.model.predict(img)
        print(f"Predicted class: {prediction[0]}")

def main():
    config = Config.load_config()

    parser = argparse.ArgumentParser(description="Предикт нейронной сети")
    parser.add_argument("--image_path", type=str, required=True, help="Путь до изображения.")
    args = parser.parse_args()

    predictor = Predictor(config)
    predictor.predict(args.image_path)

if __name__ == "__main__":
    main()
