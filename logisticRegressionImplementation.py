import ast
import re
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

    def train(self, dataset):
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
        self.index = 0 if self.index == self.get_size()-1 else self.index + 1
        return sample

    def get_samples(self, batch_size = 1):
        return [self.get_sample() for _ in batch_size]

    def get_all_samples(self):
        return self.xs, self.ys

    def union(self, other):
        other.xs, other.ys = other.get_all_samples()
        self.xs += other.xs
        self.ys += other.ys

    def compute_average_loss(self, model, step = 1):
        return np.average([model.loss(self.xs[i], self.ys[i]) for i in range(0, self.get_size(), step)])

    def compute_average_accuracy(self, model, step = 1):
        return np.average([1 if model.predict(self.xs[i]) == self.ys[i] else 0 for i in range(0, self.get_size(), step)])
    

class TrainBiasDataset(Dataset):
    """
    Dataset used for logistic regression.
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
            if ind == 0: self.feature_size = len(X)
            y = data.iloc[ind]['bias_score']
            split_data += [(X, y)]

        self.xs, self.ys = zip(*split_data)

        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size=0.2, random_state = 42)

        self.xs = X_train
        self.ys = y_train
        self.index = 0
        self.classes = classes
        self.class_labels = [str(k) for k in self.classes]
        

    def plot_accuracy_curve(self, eval_iters, accuracies, title = None):
        plt.plot(eval_iters, accuracies)
        plt.ylim([0,1])
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_confusion_matrix(self, model, step = 1, show_diagonal = False):
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
        ax.tick_params(top = True, labeltop = True, bottom = False, labelbottom = False)
        plt.xlabel("Predicted class")
        ax.xaxis.set_label_position('top')
        plt.ylabel("Actual class")
        plt.title("Confusion matrix " + ("(including diagonal)" if show_diagonal else "(off-diagonal only)"))
        plt.colorbar()
        plt.show()

class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 
    28x28 = 784 for MNIST). x is a 2-D image, represented as a list of lists (28x28 for MNIST). 
    y is an integer between 1 and num_classes. The goal is to fit P(y = k | x) = hypothesis(x)[k], 
    where hypothesis is a discrete distribution (list of probabilities) over the K classes, and 
    to make a class prediction using the hypothesis. Answers the questions from Question 5 on 
    Programming Assignment 4.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        '''
        Initializes the MultiLogisticRegressionModel class which takes in the number of features, the 
        number of classes and a learning rate to define the convergence speed of the descent. 
        Creates a 2D array of zeros that is equal to the number of features squared by the number of 
        classes to allow for a column that relates to each specific targets pixel weights. 
        Creates two accuracy arrays that will be tracked during iterations in the gradient descent to 
        track how well the model fits with different numbers of random samples from the test and train sets.
        '''
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_accuracies = []
        self.weights = np.zeros((self.num_features, self.num_classes))

    def get_features(self, x):
        '''
        Returns the features that are created based on the incoming data (the _x_ coordinate).
        This becomes a 1D array from the incoming 2D data. This will use all of the pixels of the
        imcoming image as features.
        '''
        features = np.array(x)
        return list(features)

    def get_weights(self):
        '''
        Returns the weights that have been calculated after training the model. This 
        is initialized as a 2D array of zeros. 
        '''
        return self.weights
    
    def get_training_accuracies(self):
        return self.train_accuracies

    def hypothesis(self, x):
        '''
        Starts by creating the dot product of the features and the weights. This is equaivalent to 
        the sum of all of the features multiplied by all of the weights in the Multinomial regression 
        model. After taking the dot product, the exp of the logit found is taken and then divided by
        its own sum to return a list of probabilites.
        '''
        features = self.get_features(x)
        logit = np.dot(features, self.weights)
        exp_logit = np.exp(logit - np.max(logit)) # for numerical stability
        prediction = exp_logit / np.sum(exp_logit)
        return prediction

    def predict(self, x):
        '''
        Returns the evaluation of the hypothesis, which takes the argmax of all the probabilites 
        found by the hypothesis. In this case, the probabilities are indexed from 1 to 10, so the 
        modulo of the argmax is taken such that class 10 is now identified properly as class 0. 
        '''
        probabilities = self.hypothesis(x)
        return (np.argmax(probabilities) + 1) % self.num_classes

    def loss(self, x, y):
        '''
        Complutes how far off the hypothesis based on _x_ is off from a test sample _y_. This is the 
        cross entropy loss of the model, a measure of the discrepancy between the data and 
        an estimation model.
        '''
        probabilities = self.hypothesis(x)
        correct_class_prob = probabilities[y - 1]
        return -np.log(correct_class_prob)

    def gradient(self, x, y):
        '''
        Returns a list of partial derivatives of the loss function with respect to each weight
        evaluated at the sample (x, y) and the current weights (run in hypothesis).
        '''
        features = self.get_features(x)
        probabilities = self.hypothesis(x)
        grad = np.outer(features, probabilities)
        grad[:, y - 1] -= features
        return grad

    def train(self, dataset, sample_size, accuracy_sample_frequency):
        '''
        For 60,000 samples sampled from the _dataset_, updates the weights array that will be used 
        to make predictions from the model. Computes loss after each 1000 samples for accuracy testing
        as sample size increases. The weights are updated based on the value calculated in the gradient
        based on the sample. Both accuracy for training and testing are tested in interval. 
        '''
        for i in range(sample_size):
            if (i%accuracy_sample_frequency == 0): 
                self.train_accuracies.append(dataset.compute_average_accuracy(self))
            x, y = dataset.get_sample()
            gradient = self.gradient(x, y)
            self.weights -= self.learning_rate * gradient

def bias_classification():
    # tests classification on 
    for n in [2,3]:
        # trains the model passing in both test and train sets to make sure that accuracies are tracked
        data_path = "normalized_data_WITH_CENTER.csv" if n == 3 else "normalized_data_no_center.csv"
        train_data = TrainBiasDataset(data_path = data_path, classes = n)
        feature_size = train_data.feature_size
        train_model = MultiLogisticRegressionModel(num_features=feature_size, num_classes = n)
        accuracy_sample_frequency = 250
        sample_size = 5000
        train_model.train(train_data, sample_size, accuracy_sample_frequency)
        # report the accuracies gathered during the training of the model
        accuracies = train_model.get_training_accuracies()
        print('Accuracies: {}'.format(accuracies))
        train_data.plot_accuracy_curve(eval_iters=range(0, sample_size, accuracy_sample_frequency), accuracies=accuracies, title='Training Accuracy Curve')
        train_data.plot_confusion_matrix(train_model)

def main():
    bias_classification()

if __name__ == "__main__":
    main()