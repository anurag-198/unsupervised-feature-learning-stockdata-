
import struct
import array
import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot

from utils import dataset

###########################################################################################
""" The Sparse Autoencoder class """


class SparseAutoencoder(object):
    #######################################################################################
    """ Initialization of Autoencoder object """

    def _init_(self, numberofInputUnits, numberofHiddenUnits, zeta, weightDecayParameter, SparsityPenaltyTerm):
        """ Initialize parameters of the Autoencoder object """

        self.numberofInputUnits = numberofInputUnits  # number of input units
        self.numberofHiddenUnits = numberofHiddenUnits  # number of hidden units
        self.zeta = zeta  # desired average activation of hidden units
        self.weightDecayParameter = weightDecayParameter  # weight decay parameter
        self.SparsityPenaltyTerm = SparsityPenaltyTerm  # weight of sparsity penalty term

        """ Set limits for accessing 'theta' values """

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

        hiddenlayer = self.sigmoidFunction(numpy.dot(weight1, input) + biasValue1)
        outputlayer = self.sigmoidFunction(numpy.dot(weight2, hiddenlayer) + biasValue2)

        """ Estimate the average activation value of the hidden layers """

        rhocap = numpy.sum(hiddenlayer, axis=1) / input.shape[1]

        """ Compute intermediate difference values using Backpropagation algorithm """

        intermediateDiff = outputlayer - input

        sumofsquareserror = 0.5 * numpy.sum(numpy.multiply(intermediateDiff, intermediateDiff)) / input.shape[1]
        weightdecay = 0.5 * self.weightDecayParameter * (numpy.sum(numpy.multiply(weight1, weight1)) +
                                           numpy.sum(numpy.multiply(weight2, weight2)))
        KLdivergence = self.SparsityPenaltyTerm * numpy.sum(self.zeta * numpy.log(self.zeta / rhocap) +
                                              (1 - self.zeta) * numpy.log((1 - self.zeta) / (1 - rhocap)))
        cost = sumofsquareserror + weightdecay + KLdivergence

        KLdivgrad = self.SparsityPenaltyTerm * (-(self.zeta / rhocap) + ((1 - self.zeta) / (1 - rhocap)))

        delout = numpy.multiply(intermediateDiff, numpy.multiply(outputlayer, 1 - outputlayer))
        delhid = numpy.multiply(numpy.dot(numpy.transpose(weight2), delout) + numpy.transpose(numpy.matrix(KLdivgrad)),
                                 numpy.multiply(hiddenlayer, 1 - hiddenlayer))

        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """

        W1grad = numpy.dot(delhid, numpy.transpose(input))
        W2grad = numpy.dot(delout, numpy.transpose(hiddenlayer))
        b1grad = numpy.sum(delhid, axis=1)
        b2grad = numpy.sum(delout, axis=1)

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

        thetagrad = numpy.concatenate((W1grad.flatten(), W2grad.flatten(),
                                        b1grad.flatten(), b2grad.flatten()))

        return [cost, thetagrad]


###########################################################################################
""" The Softmax Regression class """


class SoftmaxRegression(object):
    #######################################################################################
    """ Initialization of Regressor object """

    def init(self, inputsize, numclasses, weightDecayParameter):
        """ Initialize parameters of the Regressor object """

        self.inputsize = inputsize  # input vector size
        self.numclasses = numclasses  # number of classes
        self.weightDecayParameter = weightDecayParameter  # weight decay parameter

        """ Randomly initialize the class weights """

        rand = numpy.random.RandomState(int(time.time()))

        self.theta = 0.005 * numpy.asarray(rand.normal(size=(numclasses * inputsize, 1)))

    #######################################################################################


    def getGroundTruth(self, labels):
        """ Prepare data needed to construct groundtruth matrix """

        labels = numpy.array(labels).flatten()
        data = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels) + 1)



        groundtruth = scipy.sparse.csrmatrix((data, labels, indptr))
        groundtruth = numpy.transpose(groundtruth.todense())

        return groundtruth

    #######################################################################################


    def softmaxCost(self, theta, input, labels):
        """ Compute the groundtruth matrix """

        groundtruth = self.getGroundTruth(labels)

        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.numclasses, self.inputsize)

        """ Compute the class probabilities for each example """

        thetax = numpy.dot(theta, input)
        hypothesis = numpy.exp(thetax)
        probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

        """ Compute the traditional cost term """

        costexamples = numpy.multiply(groundtruth, numpy.log(probabilities))
        traditionalcost = -(numpy.sum(costexamples) / input.shape[1])

        """ Compute the weight decay term """

        thetasquared = numpy.multiply(theta, theta)
        weightdecay = 0.5 * self.weightDecayParameter * numpy.sum(thetasquared)

        """ Add both terms to get the cost """

        cost = traditionalcost + weightdecay

        """ Compute and unroll 'theta' gradient """

        thetagrad = -numpy.dot(groundtruth - probabilities, numpy.transpose(input))
        thetagrad = thetagrad / input.shape[1] + self.weightDecayParameter * theta
        thetagrad = numpy.array(thetagrad)
        thetagrad = thetagrad.flatten()

        return [cost, thetagrad]

    #######################################################################################
    """ Returns predicted classes for a set of inputs """

    def softmaxPredict(self, theta, input):
        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.numclasses, self.inputsize)

        """ Compute the class probabilities for each example """

        thetax = numpy.dot(theta, input)
        hypothesis = numpy.exp(thetax)
        probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

        """ Give the predictions based on probability values """

        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis=0)

        return predictions




###########################################################################################
""" Visualizes the obtained optimal weight1 values as images """


def visualizeW1(optW1, vispatchside, hidpatchside):
    """ Add the weights as a matrix of images """

    figure, axes = matplotlib.pyplot.subplots(nrows=hidpatchside,
                                              ncols=hidpatchside)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(optW1[index, :].reshape(vispatchside, vispatchside),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.setframeon(False)
        axis.setaxisoff()
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

    hiddenlayer = 1 / (1 + numpy.exp(-(numpy.dot(weight1, input) + biasValue1)))

    return hiddenlayer


###########################################################################################
""" Loads data, trains the Autoencoder and Regressor, tests the accuracy """

def theta():
    val = []
    with open("test.txt", "randomValues") as f:
        val = f.read()
        no = val.split("\n")

        #print(no[len(no) - 1])
        list = numpy.zeros(len(no), dtype=float)
        i = 0
        for i in range(len(no) - 1):
            list[i] = float(no[i])
        #print(list[i])
    return  list

def selfTaughtLearning():
    """ Define the parameters of the Autoencoder """

    vispatchside = 50  # side length of sampled image patches
    hidpatchside = 25  # side length of representative image patches ------------------to change here --------------------------------
    #hidpatchside = 16
    zeta = 0.1  # desired average activation of hidden units
    weightDecayParameter = 0.001  # weight decay parameter
    SparsityPenaltyTerm = 3  # weight of sparsity penalty term
    maxiterations = 800  # number of optimization iterations
    numberofInputUnits = vispatchside * vispatchside  # number of input units
    numberofHiddenUnits = hidpatchside * hidpatchside  # number of hidden units

    data,labels = dataset("gray503")

    opttheta = theta()
    #opttheta = optsolution.x
    #optW1 = opttheta[0:16*16*2500].reshape(numberofHiddenUnits, numberofInputUnits)
    optW1 = opttheta[0:25 * 25 * 2500].reshape(numberofHiddenUnits, numberofInputUnits)

    print(len(optW1))
    """ Visualize the obtained optimal weight1 weights """

    visualizeW1(optW1, vispatchside, hidpatchside)

    softmaxdata = data
    softmaxlabels = labels

    limit = int(softmaxdata.shape[1] / 5)
    testdata = softmaxdata[:, :limit]
    traindata = softmaxdata[:, limit:]
    testlabels = softmaxlabels[:limit, :]
    trainlabels = softmaxlabels[limit:, :]

    #""" Obtain training and testing features from the trained Autoencoder """

    trainfeatures = feedForwardAutoencoder(opttheta, numberofHiddenUnits, numberofInputUnits, traindata)
    testfeatures = feedForwardAutoencoder(opttheta, numberofHiddenUnits, numberofInputUnits, testdata)

    #""" Initialize parameters of the Regressor """

    inputsize = 625  # input vector size                     #---------------------------to change here ---------------------#

    #inputsize = 256
    numclasses = 3  # number of classes
    weightDecayParameter = 0.0001  # weight decay parameter
    maxiterations = 800  # number of optimization iterations

    regressor = SoftmaxRegression(inputsize, numclasses, weightDecayParameter)

    #""" Run the optimizer to get the optimal parameter values """

    optsolution = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                           args=(trainfeatures, trainlabels,), method='L-BFGS-B',
                                           jac=True, options={'maxiter': maxiterations})
    opttheta = optsolution.x

    #""" Obtain predictions from the trained model """

    predictions = regressor.softmaxPredict(opttheta, testfeatures)

    #""" Print accuracy of the trained model """

    correct = testlabels[:, 0] == predictions[:, 0]
    print("""Accuracy :""", numpy.mean(correct))


selfTaughtLearning()
