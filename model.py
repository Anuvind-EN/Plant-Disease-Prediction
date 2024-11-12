import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import matplotlib.pyplot as plt

# Dataset Class for Loading and Preprocessing
class Dataset(tf.keras.utils.Sequence):  # Ensure this inherits from Sequence
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)  # Ensure base class constructor is called
        self.labels, self.images = self.load_data()

    def load_data(self):
        labels = {}
        images = {}
        count = 0
        main_dir = os.listdir(os.path.join("dataset", "train"))
        reference = {}
        for i, dir in enumerate(main_dir):
            reference[dir] = i
            images_list = os.listdir(os.path.join("dataset", "train", dir))
            local_cnt = 0
            for img in images_list:
                if local_cnt < 500:
                    labels[count] = i
                    img_path = os.path.join("dataset", "train", dir, img)
                    img = image.load_img(img_path, target_size=(256, 256))
                    img = image.img_to_array(img)
                    images[count] = img / 255.0
                    count += 1
                    local_cnt += 1
                else:
                    break
        print(reference)
        return labels, images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_data(self):
        images = np.array(list(self.images.values()))
        labels = np.array(list(self.labels.values()))
        return images, labels

class ValDataset(Dataset):

    def load_data(self):
        labels = {}
        images = {}
        count = 0
        main_dir = os.listdir(os.path.join("dataset", "valid"))
        for i, dir in enumerate(main_dir):
            print(i, dir)
            images_list = os.listdir(os.path.join("dataset", "valid", dir))
            local_cnt = 0
            for img in images_list:
                if local_cnt < 100:
                    labels[count] = i
                    img_path = os.path.join("dataset", "valid", dir, img)
                    img = image.load_img(img_path, target_size=(256, 256))
                    img = image.img_to_array(img)
                    images[count] = img / 255.0
                    count += 1
                    local_cnt += 1
                else:
                    break
        return labels, images

# Model Architecture in TensorFlow/Keras
class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        # Increased number of filters
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        
        self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D((2, 2))

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)  # Dropout to prevent overfitting
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(17, activation='softmax')  # Assuming 17 categories

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(inputs)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.out(x)

# In the main code, where the model is created:
def train(dataset, valdataset, model):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Prepare the data
    train_images, train_labels = dataset.get_data()
    val_images, val_labels = valdataset.get_data()

    # Train the model
    history = model.fit(train_images, train_labels, epochs=20, batch_size=64,
                        validation_data=(val_images, val_labels))

    # Saving model after training
    model.save('model.keras')

    # Plotting accuracy and loss
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Saving Labels to JSON File
def save_labels():
    main_dir = os.listdir(os.path.join("dataset", "train"))
    reference = {}
    for i, dir in enumerate(main_dir):
        reference[dir] = i

    with open('labels.json', 'w') as json_file:
        json.dump(reference, json_file)

# Prediction Function
def predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    result_idx = np.argmax(prediction, axis=1)
    
    # Loading the labels from the reference dictionary
    with open('labels.json', 'r') as json_file:
        reference = json.load(json_file)
        
    # Get the label corresponding to the index
    for key, value in reference.items():
        if value == result_idx:
            print(f"Predicted Class: {key}")
            break

# Running the Model
if __name__ == "__main__":
    # Initialize datasets
    dataset = Dataset()
    valdataset = ValDataset()

    # Initialize and train the model
    model = Network()
    train(dataset, valdataset, model)

    # Save the labels to a JSON file
    save_labels()
