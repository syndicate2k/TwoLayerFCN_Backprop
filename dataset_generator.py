import os
import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image

class DatasetGenerator:
    def __init__(self, output_dir, num_samples):
        self.output_dir = output_dir
        self.num_samples = num_samples
    
    def generate(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        train_dir = os.path.join(self.output_dir, 'train')
        test_dir = os.path.join(self.output_dir, 'test')
        val_dir = os.path.join(self.output_dir, 'valid')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        
        self._save_images(X_train[:self.num_samples], y_train[:self.num_samples], train_dir)
        self._save_images(X_test[:self.num_samples], y_test[:self.num_samples], test_dir)
        self._save_images(X_test[self.num_samples:2*self.num_samples], y_test[self.num_samples:2*self.num_samples], val_dir)

    def _save_images(self, images, labels, directory):
        for i, (image, label) in enumerate(zip(images, labels)):
            class_dir = os.path.join(directory, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img = Image.fromarray(image)
            img.save(os.path.join(class_dir, f'image_{i}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание датасета из MNIST.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    generator = DatasetGenerator(args.output_dir, args.num_samples)
    generator.generate()