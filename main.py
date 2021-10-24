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
        names = base.target_names
        return base_class, predictions, dummy_class, names
    except AttributeError:
        return False, False, False, False


def plot_learning_curve_graph(model, epochs: int):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title(f'Learning curve - {epochs} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.show()


def plot_confusion_matrix_bar_graph(target_names: [], confus_matrix: [], epochs: int):
    labels = target_names

    values = []
    for confusion in confus_matrix:
        values.append(confusion)

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    rects = []

    for index, val in enumerate(values):
        if 0 < index < 2:
            rects.append(ax.bar(x + width, val, width, label=target_names[index]))
        elif index >= 2:
            rects.append(ax.bar(x - width, val, width, label=target_names[index]))
        else:
            rects.append(ax.bar(x, val, width, label=target_names[index]))


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Classifications')
    ax.set_title(f'Classifications by names - {epochs} epochs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects:
        ax.bar_label(rect, padding=3)

    fig.tight_layout()

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

    base_class, predictions, dummy_class, target_names = config(dataset)

    if base_class.any():

        X_training, X_test, y_training, y_test = train_test_split(
            predictions, dummy_class, test_size=test_size, random_state=random_state)

        model = Sequential()
        model.add(Dense(units=units, input_dim=len(predictions[0]), activation='softmax'))

        model.summary()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model_fit = model.fit(X_training, y_training, epochs=epochs, validation_data=(X_test, y_test))

        predictions = model.predict(X_test)
        predictions = (predictions > 0.5)

        confusion = get_confusion_matrix(y_test, predictions)

        plot_learning_curve_graph(model_fit, epochs)
        plot_confusion_matrix_bar_graph(target_names, confusion, epochs)

        _, train_accuracy = model.evaluate(X_training, y_training, verbose=0)
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        return train_accuracy, test_accuracy

    else:
        return print('Invalid dataset method name')


if __name__ == '__main__':
    models = []

    # 3 neurons and 2500 epochs
    #model = run_mlp_one_hidden_layer(epochs=5000)
    #models.append([model[0], model[1], 5000])

    # 3 neurons and 2500 epochs
    #model = run_mlp_one_hidden_layer(epochs=2500)
    #models.append([model[0], model[1], 2500])

    # 3 neurons and 1000 epochs
    #model = run_mlp_one_hidden_layer()
    #models.append([model[0], model[1], 1000])

    # 3 neurons and 500 epochs
    model = run_mlp_one_hidden_layer(epochs=500)
    models.append([model[0], model[1], 500])

    # 3 neurons and 250 epochs
    model = run_mlp_one_hidden_layer(epochs=250)
    models.append([model[0], model[1], 250])
