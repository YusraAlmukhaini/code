

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

def load_and_split_data():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split the data into training and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def convert_to_categorical(y_train, y_test):
    # convert y_train and y_test to categorical format
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test

def build_model():
    # Build the model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

def compile_and_train_model(model, X_train, y_train):
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

X_train, X_test, y_train, y_test = load_and_split_data()
X_train, X_test = scale_data(X_train, X_test)
y_train, y_test = convert_to_categorical(y_train, y_test)
model = build_model()
model = compile_and_train_model(model, X_train, y_train)
evaluate_model(model, X_test, y_test)



