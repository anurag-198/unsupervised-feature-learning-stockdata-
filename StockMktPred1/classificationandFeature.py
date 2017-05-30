import struct
import array
import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys
from pprint import pprint

from utils import dataset

def theta():
    val = []
    with open("testfor16*16.txt", "r") as f:
        val = f.read()
        no = val.split("\n")

        #print(no[len(no) - 1])
        list = numpy.zeros(len(no), dtype=float)
        i = 0
        for i in range(len(no) - 1):
            list[i] = float(no[i])
        #print(list[i])
    return  list

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

def visualizeW1(opt_W1, visiblePatchSide, hiddenPatchSide):
    """ Add the weights as a matrix of images """


    figure, axes = matplotlib.pyplot.subplots(nrows=hiddenPatchSide,
                                              ncols=hiddenPatchSide)

    #figure, axes = matplotlib.pyplot.subplots(nrows=2,
    #                                          ncols=5)

    index = 0
    #print (axes)
    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(visiblePatchSide, visiblePatchSide),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    matplotlib.pyplot.show()


def classif():
    data, labels = dataset("gray50")
    softmax_data = data
    softmax_labels = labels

    visiblePatchSide = 50  # side length of sampled image patches
    hiddenPatchSide = 16  # side length of representative image patches
    #hiddenPatchSide = 25
    zeta = 0.1  # desired average activation of hidden units
    weightDecayParameter = 0.001  # weight decay parameter
    SparsityPenaltyTerm = 3  # weight of sparsity penalty term
    num_OptimizedIterations = 800  # number of optimization iterations
    numberofInputUnits = visiblePatchSide * visiblePatchSide  # number of input units
    numberofHiddenUnits = hiddenPatchSide * hiddenPatchSide  # number of hidden units

    # """ Split the Softmax set into two halves, one for training and one for testing """

    limit = int(softmax_data.shape[1] / 5)
    test_data = softmax_data[:, :limit]
    train_data = softmax_data[:, limit:]
    test_labels = softmax_labels[:limit, :]
    train_labels = softmax_labels[limit:, :]
    opt_theta = theta()                     # getting the stored theta obtained after training ................

    print("theta read")
    # """ Obtain training and testing features from the trained Autoencoder """
    train_features = feedForwardAutoencoder(opt_theta, numberofHiddenUnits, numberofInputUnits, train_data)
    test_features = feedForwardAutoencoder(opt_theta, numberofHiddenUnits, numberofInputUnits, test_data)


    t_f = numpy.array(train_features)
    tf = numpy.transpose(t_f)
    tesf = numpy.array(test_features)
    tsf = numpy.transpose(tesf)

    tl = numpy.array(train_labels)

    ###################################### feature ranking for reduced features ###########################

    print(t_f.shape, tl.shape)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize
    rf.fit(tf, tl)

    featureImportance = rf.feature_importances_

    n = len(featureImportance)
    x = range(n)

    ffi_pair = zip(x, featureImportance)

    ffi_pair = sorted(ffi_pair, key=lambda x: x[1])

    sol = ffi_pair[::-1]
    sol1 = sol[:10]
    sol1 = sol[-10:]
    print(sol1)
    print(sol[:10])

    #lis = [x[0] for x in sol1]
    lis = [x[0] for x in sol]
    limitZero = 0
    limitOne = numberofHiddenUnits * numberofInputUnits
    opt_W1 = opt_theta[limitZero: limitOne].reshape(numberofHiddenUnits, numberofInputUnits)

    weig = numpy.array(opt_W1)
    wt = weig[lis,:]

    print(wt.shape)

    #visualizeW1(weig, visiblePatchSide, hiddenPatchSide)
    visualizeW1(wt, visiblePatchSide, hiddenPatchSide)
    exit(0)

    print("all shapes ")
    print(tsf.shape)
    print(tl.shape)
    print(tf.shape)






  ########################################################Classification task begins here ###########################################
    # """ Initialize parameters of the Regressor """

    #input_size = 625  # input vector size

'''
    input_size = 2500     #------------------------------------------------------ for normal features comment later

    num_classes = 2  # number of classes
    weightDecayParameter = 0.0001  # weight decay parameter
    num_OptimizedIterations = 400  # number of optimization iterations

    regressor = SoftmaxRegression(input_size, num_classes, weightDecayParameter)

    # """ Run the L-BFGS algorithm to get the optimal parameter values """

    #opt_solution = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
    #                                       args=(train_features, train_labels,), method='L-BFGS-B',
    #                                       jac=True, options={'maxiter': num_OptimizedIterations})

    t_f = train_data # -----------------------------------------------------------------------------comment later
    opt_solution = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                           args=(t_f, tl,), method='L-BFGS-B',
                                           jac=True, options={'maxiter': num_OptimizedIterations})       #------------comment later
    opt_theta = opt_solution.x

    # """ Obtain predictions from the trained model """

    #predictions = regressor.softmaxPredict(opt_theta, test_features)
    predictions = regressor.softmaxPredict(opt_theta, test_data)  #--------------------------- test data normal comment later
    # """ Print accuracy of the trained model """

    correct = test_labels[:, 0] == predictions[:, 0]
    print("""Accuracy for Softmax :""", numpy.mean(correct))

    ############################SVM classifier#################################################3

    print(" applying SVM classification ")
    clff = svm.SVC()

    #print(len(train_features), len(train_labels))
    tf = numpy.transpose(t_f)
    tsf = numpy.transpose(test_data)

    print(tf.shape, tsf.shape)
    clff.fit(tf, tl)
    predict = clff.predict(tsf)

    print(len(predict), len(test_labels))

    cnt = 0
    for i in range(len(predict)) :
        if (predict[i] == test_labels[i][0]) :
            cnt = cnt + 1

    acc = cnt / len(predict)
    print(cnt, acc)

    print("""Accuracy for SVM :""", acc)

'''
classif()