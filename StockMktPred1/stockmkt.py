
import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot

from utils import dataset

class SparseAutoencoder(object):

    def __init__(self, numberofInputUnits, numberofHiddenUnits, zeta, weightDecayParameter, sparsityPenaltyTerm):
        """ Initialize parameters of the Autoencoder object """

        self.numberofInputUnits = numberofInputUnits  # number of input units
        self.numberofHiddenUnits = numberofHiddenUnits  # number of hidden units
        self.zeta = zeta  # desired average activation of hidden units
        self.weightDecayParameter = weightDecayParameter  # weight decay parameter
        self.sparsityPenaltyTerm = sparsityPenaltyTerm  # weight of sparsity penalty term

        self.limitZero = 0
        self.limitOne = numberofHiddenUnits * numberofInputUnits
        self.limitTwo = 2 * numberofHiddenUnits * numberofInputUnits
        self.limitThree = 2 * numberofHiddenUnits * numberofInputUnits + numberofHiddenUnits
        self.limitFour = 2 * numberofHiddenUnits * numberofInputUnits + numberofHiddenUnits + numberofInputUnits

        """ Initialize Neural Network weights randomly
            weight1, weight2 values are chosen in the range [-randomValues, randomValues] """

        randomValues = math.sqrt(6) / math.sqrt(numberofInputUnits + numberofHiddenUnits + 1)

        rand = numpy.random.RandomState(int(time.time()))

        weight1 = numpy.asarray(rand.uniform(low=-randomValues, high=randomValues, size=(numberofHiddenUnits, numberofInputUnits)))
        weight2 = numpy.asarray(rand.uniform(low=-randomValues, high=randomValues, size=(numberofInputUnits, numberofHiddenUnits)))

        """ Bias values are initialized to zero """

        biasValue1 = numpy.zeros((numberofHiddenUnits, 1))
        biasValue2 = numpy.zeros((numberofInputUnits, 1))

        """ Create 'theta' by unrolling weight1, weight2, biasValue1, biasValue2 """

        self.theta = numpy.concatenate((weight1.flatten(), weight2.flatten(),
                                        biasValue1.flatten(), biasValue2.flatten()))

    #######################################################################################
    """ Returns elementwise sigmoidFunction output of input array """

    def sigmoidFunction(self, x):
        return (1 / (1 + numpy.exp(-x)))

    #######################################################################################
    """ Returns gradient of 'theta' using Backpropagation """

    def sparse_autoencoder_cost(self, theta, input):
        """ Extract weights and biases from 'theta' input """

        weight1 = theta[self.limitZero: self.limitOne].reshape(self.numberofHiddenUnits, self.numberofInputUnits)
        weight2 = theta[self.limitOne: self.limitTwo].reshape(self.numberofInputUnits, self.numberofHiddenUnits)
        biasValue1 = theta[self.limitTwo: self.limitThree].reshape(self.numberofHiddenUnits, 1)
        biasValue2 = theta[self.limitThree: self.limitFour].reshape(self.numberofInputUnits, 1)

        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """

        hidden_layer = self.sigmoidFunction(numpy.dot(weight1, input) + biasValue1)
        output_layer = self.sigmoidFunction(numpy.dot(weight2, hidden_layer) + biasValue2)

        """ Estimate the average activation value of the hidden layers """

        avgActivationValue = numpy.sum(hidden_layer, axis=1) / input.shape[1]

        """ Compute intermediate difference values using Backpropagation algorithm """

        intermediateDiff = output_layer - input

        sumofsquaresError = 0.5 * numpy.sum(numpy.multiply(intermediateDiff, intermediateDiff)) / input.shape[1]
        wtDecay = 0.5 * self.weightDecayParameter * (numpy.sum(numpy.multiply(weight1, weight1)) +
                                           numpy.sum(numpy.multiply(weight2, weight2)))
        KL_divergence = self.sparsityPenaltyTerm * numpy.sum(self.zeta * numpy.log(self.zeta / avgActivationValue) +
                                              (1 - self.zeta) * numpy.log((1 - self.zeta) / (1 - avgActivationValue)))
        cost = sumofsquaresError + wtDecay + KL_divergence

        KL_div_grad = self.sparsityPenaltyTerm * (-(self.zeta / avgActivationValue) + ((1 - self.zeta) / (1 - avgActivationValue)))

        del_out = numpy.multiply(intermediateDiff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(weight2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)),
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))

        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """

        W1grad = numpy.dot(del_hid, numpy.transpose(input))
        W2grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1grad = numpy.sum(del_hid, axis=1)
        b2grad = numpy.sum(del_out, axis=1)

        W1grad = W1grad / input.shape[1] + self.weightDecayParameter * weight1
        W2grad = W2grad / input.shape[1] + self.weightDecayParameter * weight2
        b1grad = b1grad / input.shape[1]
        b2grad = b2grad / input.shape[1]

        """ Transform numpy matrices into arrays """

        W1grad = numpy.array(W1grad)
        W2grad = numpy.array(W2grad)
        b1grad = numpy.array(b1grad)
        b2grad = numpy.array(b2grad)

        """ Unroll the gradient values and return as 'theta' gradient """

        thetaGrad = numpy.concatenate((W1grad.flatten(), W2grad.flatten(),
                                        b1grad.flatten(), b2grad.flatten()))

        return [cost, thetaGrad]


###########################################################################################
""" The Softmax Regression class """


class SoftmaxRegression(object):
    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, weightDecayParameter):
        """ Initialize parameters of the Regressor object """

        self.input_size = input_size  # input vector size
        self.num_classes = num_classes  # number of classes
        self.weightDecayParameter = weightDecayParameter  # weight decay parameter

        """ Randomly initialize the class weights """

        rand = numpy.random.RandomState(int(time.time()))

        self.theta = 0.005 * numpy.asarray(rand.normal(size=(num_classes * input_size, 1)))

    #######################################################################################


    def getGroundTruth(self, labels):
        """ Prepare data needed to construct groundtruth matrix """

        labels = numpy.array(labels).flatten()
        data = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels) + 1)
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = numpy.transpose(ground_truth.todense())

        return ground_truth

    #######################################################################################


    def softmaxCost(self, theta, input, labels):
        """ Compute the groundtruth matrix """

        ground_truth = self.getGroundTruth(labels)

        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.num_classes, self.input_size)

        """ Compute the class probabilities for each example """

        theta_x = numpy.dot(theta, input)
        hypothesis = numpy.exp(theta_x)
        probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

        """ Compute the traditional cost term """

        costExamples = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(costExamples) / input.shape[1])

        """ Compute the weight decay term """

        theta_squared = numpy.multiply(theta, theta)
        wtDecay = 0.5 * self.weightDecayParameter * numpy.sum(theta_squared)

        """ Add both terms to get the cost """

        cost = traditional_cost + wtDecay

        """ Compute and unroll 'theta' gradient """

        thetaGrad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        thetaGrad = thetaGrad / input.shape[1] + self.weightDecayParameter * theta
        thetaGrad = numpy.array(thetaGrad)
        thetaGrad = thetaGrad.flatten()

        return [cost, thetaGrad]

    #######################################################################################
    """ Returns predicted classes for a set of inputs """

    def softmaxPredict(self, theta, input):
        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.num_classes, self.input_size)

        """ Compute the class probabilities for each example """

        theta_x = numpy.dot(theta, input)
        hypothesis = numpy.exp(theta_x)
        probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

        """ Give the predictions based on probability values """

        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis=0)

        return predictions


###########################################################################################
""" Visualizes the obtained optimal weight1 values as images """


def visualizeW1(opt_W1, visiblePatchSide, hiddenPatchSide):

    """ Add the weights as a matrix of images """

    figure, axes = matplotlib.pyplot.subplots(nrows=hiddenPatchSide,
                                              ncols=hiddenPatchSide)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(visiblePatchSide, visiblePatchSide),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    matplotlib.pyplot.show()


###########################################################################################
""" Returns the hidden layer activations of the Autoencoder """


def feedForwardAutoencoder(theta, numberofHiddenUnits, numberofInputUnits, input):
    """ Define limits to access useful data """

    limitZero = 0
    limitOne = numberofHiddenUnits * numberofInputUnits
    limitTwo = 2 * numberofHiddenUnits * numberofInputUnits
    limitThree = 2 * numberofHiddenUnits * numberofInputUnits + numberofHiddenUnits

    """ Access weight1 and biasValue1 from 'theta' """

    weight1 = theta[limitZero: limitOne].reshape(numberofHiddenUnits, numberofInputUnits)
    biasValue1 = theta[limitTwo: limitThree].reshape(numberofHiddenUnits, 1)

    """ Compute the hidden layer activations """

    hidden_layer = 1 / (1 + numpy.exp(-(numpy.dot(weight1, input) + biasValue1)))

    return hidden_layer



###########################################################################################
""" Loads data, trains the Autoencoder and Regressor, tests the accuracy """


def selfTaughtLearning():
    """ Define the parameters of the Autoencoder """

    visiblePatchSide = 50  # side length of sampled image patches
    #hiddenPatchSide = 25  # side length of representative image patches ------------------to change here --------------------------------
    hiddenPatchSide = 16
    zeta = 0.1  # desired average activation of hidden units
    weightDecayParameter = 0.001  # weight decay parameter
    sparsityPenaltyTerm = 3  # weight of sparsity penalty term
    num_OptimizedIterations = 800  # number of optimization iterations
    numberofInputUnits = visiblePatchSide * visiblePatchSide  # number of input units
    numberofHiddenUnits = hiddenPatchSide * hiddenPatchSide  # number of hidden units

    """ Load MNIST images for training and testing """

    data,labels = dataset("gray503")



    """ Initialize the Autoencoder with the above parameters """

    encoder = SparseAutoencoder(numberofInputUnits, numberofHiddenUnits, zeta, weightDecayParameter, sparsityPenaltyTerm)

    print("running the optimization process -- ")
    """ Run the optimizer to get the optimal parameter values """

    opt_solution = scipy.optimize.minimize(encoder.sparse_autoencoder_cost, encoder.theta,
                                           args=(data,), method='L-BFGS-B',
                                           jac=True, options={'maxiter': num_OptimizedIterations})
    opt_theta = opt_solution.x
    opt_W1 = opt_theta[encoder.limitZero: encoder.limitOne].reshape(numberofHiddenUnits, numberofInputUnits)

    print(len(opt_W1))
    """ Visualize the obtained optimal weight1 weights """

    visualizeW1(opt_W1, visiblePatchSide, hiddenPatchSide)
    #numpy.savetxt('testfor16*16.txt', opt_theta, delimiter=',')

    #""" Assign data of digits 0-4 to Softmax """

    #softmax_set = numpy.array(((labels >= 0) & (labels <= 4)).flatten())
    softmax_data = data
    softmax_labels = labels

    #""" Split the Softmax set into two halves, one for training and one for testing """

    limit = int(softmax_data.shape[1] / 5)
    test_data = softmax_data[:, :limit]
    train_data = softmax_data[:, limit:]
    test_labels = softmax_labels[:limit, :]
    train_labels = softmax_labels[limit:, :]

    #""" Obtain training and testing features from the trained Autoencoder """

    train_features = feedForwardAutoencoder(opt_theta, numberofHiddenUnits, numberofInputUnits, train_data)
    test_features = feedForwardAutoencoder(opt_theta, numberofHiddenUnits, numberofInputUnits, test_data)

    #""" Initialize parameters of the Regressor """

    #input_size = 625  # input vector size                     #---------------------------to change here ---------------------#

    input_size = 196
    num_classes = 3  # number of classes
    weightDecayParameter = 0.0001  # weight decay parameter
    num_OptimizedIterations = 800  # number of optimization iterations

    regressor = SoftmaxRegression(input_size, num_classes, weightDecayParameter)

    #""" Run the optimizer to get the optimal parameter values """

    opt_solution = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                           args=(train_features, train_labels,), method='L-BFGS-B',
                                           jac=True, options={'maxiter': num_OptimizedIterations})
    opt_theta = opt_solution.x

    #""" Obtain predictions from the trained model """

    predictions = regressor.softmaxPredict(opt_theta, test_features)

    #""" Print accuracy of the trained model """

    correct = test_labels[:, 0] == predictions[:, 0]
    print("""Accuracy :""", numpy.mean(correct))


selfTaughtLearning()
