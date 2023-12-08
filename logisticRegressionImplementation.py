import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Model:
    """
    Abstract class for a machine learning model.
    """

    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset, sample_size, accuracy_sample_frequency):
        pass


class Dataset:
    """
    Abstract class for a machine learning dataset.
    """

    def __init__(self):
        self.xs = [None]
        self.ys = [None]
        self.index = 0

    def get_size(self):
        return len(self.xs)

    def get_sample(self):
        sample = (self.xs[self.index], self.ys[self.index])
        self.index = 0 if self.index == self.get_size() - 1 else self.index + 1
        return sample

    def get_samples(self, batch_size=1):
        return [self.get_sample() for _ in batch_size]

    def get_all_samples(self):
        return self.xs, self.ys

    def union(self, other):
        other.xs, other.ys = other.get_all_samples()
        self.xs += other.xs
        self.ys += other.ys

    def compute_average_loss(self, model, step=1):
        return np.average([model.loss(self.xs[i], self.ys[i]) for i in range(0, self.get_size(), step)])

    def compute_average_accuracy(self, model, step=1):
        return np.average(
            [1 if model.predict(self.xs[i]) == self.ys[i] else 0 for i in range(0, self.get_size(), step)])


class TrainBiasDataset(Dataset):
    """
    Dataset specifically tailored for logistic regression. It initializes with data from a specified path
    and the number of classes. The class handles data preprocessing, splitting it into training and testing
    sets. Notable methods include plot_accuracy_curve, which visualizes the training accuracy curve over
    iterations, and plot_confusion_matrix, which plots the confusion matrix of the model's predictions.
    The class provides a comprehensive set of tools for evaluating and visualizing the performance of logistic
    regression models on bias classification tasks.
    """

    def __init__(self, data_path, classes):

        data = pd.read_csv(data_path)
        classes = range(classes)

        split_data = []
        self.feature_size = 0

        for ind in data.index:
            p_vectors = data.iloc[ind][[str(i) for i in range(100)]].to_numpy()
            X = data.iloc[ind][['topic', 'source']].to_numpy()
            X = np.concatenate((np.array(X), np.array(p_vectors)))
            if ind == 0:
                self.feature_size = len(X)
            y = data.iloc[ind]['bias_score']
            split_data += [(X, y)]

        self.xs, self.ys = zip(*split_data)

        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size=0.2, random_state=42)

        self.xs = X_train
        self.ys = y_train
        self.index = 0
        self.classes = classes
        self.class_labels = [str(k) for k in self.classes]

    def plot_accuracy_curve(self, eval_iters, accuracies, title=None):
        plt.plot(eval_iters, accuracies)
        plt.ylim([0, 1])
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_confusion_matrix(self, model, step=1, show_diagonal=False):
        confusion = [[0] * len(self.classes) for _ in self.classes]
        num_correct = 0
        num_evaluated = 0
        for i in range(0, self.get_size(), step):
            class_prediction = model.predict(self.xs[i])
            class_actual = self.ys[i]
            num_correct += 1 if class_prediction == class_actual else 0
            num_evaluated += 1
            confusion[class_actual][class_prediction] += 1
        print("Accuracy:", num_correct / num_evaluated, '%  (', num_correct, 'out of', num_evaluated, ')')

        print("Confusion matrix:")
        for k in range(len(self.classes)):
            print(self.class_labels[k], confusion[k])

        confusion_plot = confusion
        if not show_diagonal:
            for k in range(len(self.classes)):
                confusion_plot[k][k] = 0
        plt.imshow(confusion_plot)
        ax = plt.gca()
        ax.set_xticks(range(len(self.classes)), self.class_labels)
        ax.set_yticks(range(len(self.classes)), self.class_labels)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.xlabel("Predicted class")
        ax.xaxis.set_label_position('top')
        plt.ylabel("Actual class")
        plt.title("Confusion matrix " + ("(including diagonal)" if show_diagonal else "(off-diagonal only)"))
        plt.colorbar()
        plt.show()


class MultiLogisticRegressionModel(Model):
    """
    The MultiLogisticRegressionModel class is a concrete implementation of the abstract Model class,
    representing a multi-class logistic regression model. It is initialized with parameters such as the
    number of features, classes, and learning rate. The class includes methods for extracting features,
    retrieving weights, computing the hypothesis, making predictions, calculating the loss, computing the
    gradient, and training the model. These methods collectively define the behavior of a logistic regression
    model and are critical for its functionality in classification tasks.
    """

    def __init__(self, num_features, num_classes, learning_rate=1e-2):
        """
        This method initializes an instance of the MultiLogisticRegressionModel class. It takes parameters
        for the number of features, number of classes, and an optional learning rate. The method sets up
        the model with zero-initialized weights, the specified number of features, classes, and learning rate.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_accuracies = []
        self.weights = np.zeros((self.num_features, self.num_classes))

    def get_features(self, x):
        """
        The get_features method converts the input x into a numpy array and returns it. This method is
        responsible for processing input features and is used internally in other methods of the class.
        """
        features = np.array(x)
        return list(features)

    def get_weights(self):
        """
        The get_weights method returns the current weights of the model. These weights are the parameters that
        the model learns during training and are essential for making predictions.
        """
        return self.weights

    def get_training_accuracies(self):
        """
        This method returns the list of training accuracies recorded during the training process. The accuracies
        are sampled at regular intervals during training to track the model's performance.
        """
        return self.train_accuracies

    def hypothesis(self, x):
        """
        The hypothesis method calculates the predicted probabilities for each class based on the input features x.
        It performs a dot product between the features and the model weights, applies a softmax function for
        normalization, and returns the resulting probability distribution.
        """
        features = self.get_features(x)
        logit = np.dot(features, self.weights)
        exp_logit = np.exp(logit - np.max(logit))
        prediction = exp_logit / np.sum(exp_logit)
        return prediction

    def predict(self, x):
        """
        The predict method uses the model's hypothesis to make a class prediction for the input features x. It
        returns the index of the class with the highest predicted probability.
        """
        probabilities = self.hypothesis(x)
        return (np.argmax(probabilities) + 1) % self.num_classes

    def loss(self, x, y):
        """
        The loss method calculates the negative log-likelihood loss for a given input x and the true class label y.
        It evaluates how well the model's predictions match the actual class labels.
        """
        probabilities = self.hypothesis(x)
        correct_class_prob = probabilities[y - 1]
        return -np.log(correct_class_prob)

    def gradient(self, x, y):
        """
        The gradient method computes the gradient of the negative log-likelihood loss with respect to the model's
        weights. This gradient is used in the training process to update the model parameters and improve its
        performance.
        """
        features = self.get_features(x)
        probabilities = self.hypothesis(x)
        grad = np.outer(features, probabilities)
        grad[:, y - 1] -= features
        return grad

    def train(self, dataset, sample_size, accuracy_sample_frequency):
        """
        The train method trains the logistic regression model using the provided dataset. It performs gradient
        descent by iteratively updating the model weights based on the computed gradients. The training process
        is controlled by the specified sample_size (number of iterations) and accuracy_sample_frequency (frequency
        at which training accuracies are recorded). Training accuracies are stored for later analysis.
        """
        for i in range(sample_size):
            if i % accuracy_sample_frequency == 0:
                accuracy = dataset.compute_average_accuracy(self)
                self.train_accuracies.append(accuracy)
                print(f'Iteration {i+250},', f'Accuracy: {accuracy}')
            x, y = dataset.get_sample()
            gradient = self.gradient(x, y)
            self.weights -= self.learning_rate * gradient


def bias_classification():
    """
    The bias_classification function serves as a testing ground for logistic regression models on bias
    classification tasks in regard to bias in articles. It iterates over different datasets, training
    models and reporting accuracies. The function initializes instances of the TrainBiasDataset and
    MultiLogisticRegressionModel classes, trains the model, and prints and plots training accuracies.
    This function encapsulates the process of evaluating logistic regression models on various bias
    classification for articles.
    """
    # Tests classification on
    for n in [2, 3]:

        # Trains the model passing in both test and train sets to make sure that accuracies are tracked
        data_path = "normalizedDataWithCenter.csv" if n == 3 else "normalizedDataNoCenter.csv"
        train_data = TrainBiasDataset(data_path=data_path, classes=n)

        # Set feature size
        feature_size = train_data.feature_size

        # Initialize training model
        train_model = MultiLogisticRegressionModel(num_features=feature_size, num_classes=n)

        # Set accuracy sample frequency
        accuracy_sample_frequency = 250

        # Set sample_size
        sample_size = 5000

        # Train model
        train_model.train(train_data, sample_size, accuracy_sample_frequency)

        # Report the accuracies gathered during the training of the model
        accuracies = train_model.get_training_accuracies()
        print(f'Accuracies: {accuracies}')

        # Plot the accuracy curve
        train_data.plot_accuracy_curve(eval_iters=range(0, sample_size, accuracy_sample_frequency),
                                       accuracies=accuracies, title='Training Accuracy Curve')

        # Plot the confusion matrix
        train_data.plot_confusion_matrix(train_model)


def main():
    bias_classification()


if __name__ == "__main__":
    main()
