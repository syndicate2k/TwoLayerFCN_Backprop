import os
import cv2
import numpy as np
import json

class Config:
    def __init__(self, input_size=784, hidden_size=256, output_size=10, learning_rate=0.01, epochs=1000, weights_path="weights/model_weights.npz"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_path = weights_path

    def save_config(self, file_path="config.json"):
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load_config(cls, file_path="config.json"):
        if not os.path.exists(file_path):
            return cls()
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

class DatasetLoader:
    @staticmethod
    def load_images_from_folder(folder, dataset_type='train', target_size=(28, 28)):
        images = []
        labels = []

        split_dir = os.path.join(folder, dataset_type)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Директории {split_dir} не существует.")

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, filename)
                    if os.path.isfile(img_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, target_size)
                            img = img.flatten().astype('float32') / 255.0
                            images.append(img)
                            labels.append(int(class_name))

        return np.array(images), np.array(labels)

    @staticmethod
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels.astype(int)]
