import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk
# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    pass

def appendOnes (images):
    return np.vstack((images.T, np.ones(images.shape[0]))).T

def rotate(imgs):
    imgsPrime = formatImg(imgs = imgs)
    return np.array([sk.rotate(image = x, angle = 15) for x in imgsPrime[0:]])
    

def scale(imgs):
    imgsPrime = formatImg(imgs = imgs)
    upscaled = np.array([sk.rescale(image = x, scale = 2) for x in imgsPrime[0:]])
    return np.array([sk.resize(x[4:-4, 4:-4], (28,28)) for x in upscaled[0:]])

def translate(imgs):
    zRow = np.zeros((1, 28)) 
    imgsPrime = formatImg(imgs = imgs)
    trimmedimgs = imgsPrime[0:, 0:-2]
    return [np.vstack((np.repeat(zRow, 2, axis = 0), x)) for x in trimmedimgs[0:]]


def applyNoise(imgs):
    imgsPrime = formatImg(imgs = imgs)
    noise = np.random.normal(loc=.5, scale=.01, size=imgsPrime.shape)
    return imgsPrime + noise

def formatImg(imgs):
    return np.array([np.resize(x, (28,28)) for x in imgs[0:]])

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...
    trans = translate(trainingImages)
    noise = applyNoise(trainingImages)
    rot = rotate(trainingImages)
    scl = scale(trainingImages)

    for x in range(1,5):
        showimg = formatImg(imgs = trainingImages)

        plt.imshow(showimg[x]), plt.show()

        plt.imshow(trans[x]), plt.show()
        plt.imshow(noise[x]), plt.show()
        plt.imshow(rot[x]), plt.show()
        plt.imshow(scl[x]), plt.show()

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)
    
    # Visualize the vectors
    # ...

