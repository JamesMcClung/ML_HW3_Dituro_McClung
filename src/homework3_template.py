import numpy as np
import matplotlib.pyplot as plt
import datetime

def shuffle_pairwise(a, b):
    '''Returns a pair of shuffled copies of the given arrays, shuffled in the same way.'''
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_predictions(W, images):
    # images: _x785
    # W: 785x10
    # intermediate layer for softmax corresponding to exp(Z)
    expZ = np.exp(images.dot(W)) # dimension: _x10
    return expZ / np.sum(expZ, axis=1).reshape(-1,1)

def get_gradient_CE(images, predictions, labels):
    # images: nx785
    # predictions: nx10
    # labels: nx10
    n = len(images)
    return images.T.dot(predictions - labels) / n # dimension: 785x10

def train_epoch_SGD(W, trainingImages, trainingLabels, epsilon, batchSize):
    '''
    Trains the weights over one epoch.
    '''
    for i in range(0, 5000, batchSize):
        images = trainingImages[i:i+batchSize]
        labels = trainingLabels[i:i+batchSize]
        predictions = get_predictions(W, images)
        W -= get_gradient_CE(images, predictions, labels) * epsilon
    return W


# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None, num_epochs=1):
    # trainingImages: 5000x785
    # trainingLabels: 5000x10
    W = np.random.normal(size=(785, 10)) # initialize weights to random numbers

    for e in range(num_epochs):
        W = train_epoch_SGD(W, trainingImages, trainingLabels, epsilon or .1, batchSize or 100)
    return W

def appendOnes (images):
    return np.vstack((images.T, np.ones(images.shape[0]))).T

def fPC(W, images, labels):
    '''Percent correct.'''
    predictions = get_predictions(W, images)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def fCE(W, images, labels):
    '''Cross-entropy loss.'''
    n = len(images)
    predictions = get_predictions(W, images)
    return -np.sum(labels * np.log(predictions)) / n

def display_weights(W):
    '''
    Displays the weights as a grayscale image.
    '''
    fig,ax = plt.subplots(3, 4)
    for i, axi in enumerate(ax.flatten()):
        if i < 10:
            img = W[:-1, i].reshape(28, 28)
            axi.imshow(img , cmap='gray')
        else:
            axi.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # shuffle data
    trainingImages, trainingLabels = shuffle_pairwise(trainingImages, trainingLabels)

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = appendOnes(trainingImages)
    testingImages = appendOnes(testingImages)

    # do regression (time it)
    start = datetime.datetime.now()
    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, num_epochs=1024)
    stop = datetime.datetime.now()
    print('Time to train: %.2f seconds' % (stop - start).total_seconds())

    # print fCE and fPC
    print(f"Test accuracy (PC): {fPC(W, testingImages, testingLabels)}")
    print(f"Test loss (CE): {fCE(W, testingImages, testingLabels)}")

    # Visualize the vectors
    display_weights(W)