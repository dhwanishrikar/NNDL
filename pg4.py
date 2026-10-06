# Lab 4 
# 21 - 08 - 2026
'''Design and implement a CNN model (with 4+ layers of convolutions) to classify multi category image datasets. Use the concept of regularization and dropout while 
designing the CNN model. Use the Fashion MNIST datasets. Record the Training accuracy and Test accuracy corresponding to the following architectures:
5. Base Model
6. Model with L1 Regularization
7. Model with L2 Regularization
8. Model with Dropout'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.regularizers import l1, l2

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and add channel dimension
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

# Function to create model
def create_model(reg=None, dropout=None):
    model = Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1), kernel_regularizer=reg),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=reg),
        layers.Flatten()
    ])

    if dropout:
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Train and test
def run(name, reg=None, dropout=None):
    print("\n---", name, "---")
    
    model = create_model(reg, dropout)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_split=0.2)
    
    loss, acc = model.evaluate(x_test, y_test)
    print("Test Accuracy:", acc * 100, "%")
    print("Loss:", loss)

# Four experiments
run("Base Model")
run("L1 Regularization", l1(0.001))
run("L2 Regularization", l2(0.001))
run("Dropout", dropout=0.5)
