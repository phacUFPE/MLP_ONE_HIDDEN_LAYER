import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

from constants import TEST_SIZE, RANDOM_STATE


def config(method_name: str = 'load_iris') -> tuple:
    try:
        load = getattr(datasets, method_name)
        base = load()
        predictions = base.data
        base_class = base.target
        dummy_class = np_utils.to_categorical(base_class)
        return base_class, predictions, dummy_class
    except AttributeError:
        return False, False, False


def plot_learning_curve_graph(model):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.show()


def get_confusion_matrix(y: [], predictions: []):
    y_test_matrix = [np.argmax(t) for t in y]
    y_prediction_matrix = [np.argmax(t) for t in predictions]

    return confusion_matrix(y_test_matrix, y_prediction_matrix)


def run_mlp_one_hidden_layer(
        units: int = 3,
        epochs: int = 1000,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        dataset: str = 'load_iris'):

    base_class, predictions, dummy_class = config(dataset)

    if not base_class:
        return print('Invalid dataset method name')

    X_training, X_test, y_training, y_test = train_test_split(
        predictions, dummy_class, test_size=test_size, random_state=random_state)

    model = Sequential()
    model.add(Dense(units=units, input_dim=len(predictions[0]), activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_trained = model.fit(X_training, y_training, epochs=epochs, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    predictions = (predictions > 0.5)
    print(f'predictions: {predictions}')

    confusion = get_confusion_matrix(y_test, predictions)
    print(f'confusion matrix: {confusion}')

    plot_learning_curve_graph(model_trained)


if __name__ == '__main__':
    run_mlp_one_hidden_layer()
