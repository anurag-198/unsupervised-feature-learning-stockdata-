
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

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
        """ Initialize parameters of the Autoencoder object """

        self.visible_size = visible_size  # number of input units
        self.hidden_size = hidden_size  # number of hidden units
        self.rho = rho  # desired average activation of hidden units
        self.lamda = lamda  # weight decay parameter
        self.beta = beta  # weight of sparsity penalty term

        """ Set limits for accessing 'theta' values """

        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size

        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """

        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)

        rand = numpy.random.RandomState(int(time.time()))

        W1 = numpy.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low=-r, high=r, size=(visible_size, hidden_size)))

        """ Bias values are initialized to zero """

        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    #######################################################################################
    """ Returns elementwise sigmoid output of input array """

    def sigmoid(self, x):
        return (1 / (1 + numpy.exp(-x)))

    #######################################################################################
    """ Returns gradient of 'theta' using Backpropagation """

    def sparseAutoencoderCost(self, theta, input):
        """ Extract weights and biases from 'theta' input """

        W1 = theta[self.limit0: self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1: self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2: self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3: self.limit4].reshape(self.visible_size, 1)

        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """

        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)

        """ Estimate the average activation value of the hidden layers """

        rho_cap = numpy.sum(hidden_layer, axis=1) / input.shape[1]

        """ Compute intermediate difference values using Backpropagation algorithm """

        diff = output_layer - input

        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                           numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                              (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))
        cost = sum_of_squares_error + weight_decay + KL_divergence

        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))

        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)),
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))

        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """

        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis=1)
        b2_grad = numpy.sum(del_out, axis=1)

        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]

        """ Transform numpy matrices into arrays """

        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)

        """ Unroll the gradient values and return as 'theta' gradient """

        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))

        return [cost, theta_grad]


###########################################################################################
""" The Softmax Regression class """


class SoftmaxRegression(object):
    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):
        """ Initialize parameters of the Regressor object """

        self.input_size = input_size  # input vector size
        self.num_classes = num_classes  # number of classes
        self.lamda = lamda  # weight decay parameter

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

        cost_examples = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])

        """ Compute the weight decay term """

        theta_squared = numpy.multiply(theta, theta)
        weight_decay = 0.5 * self.lamda * numpy.sum(theta_squared)

        """ Add both terms to get the cost """

        cost = traditional_cost + weight_decay

        """ Compute and unroll 'theta' gradient """

        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()

        return [cost, theta_grad]

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
""" Visualizes the obtained optimal W1 values as images """


def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):
    """ Add the weights as a matrix of images """

    figure, axes = matplotlib.pyplot.subplots(nrows=hid_patch_side,
                                              ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    matplotlib.pyplot.show()


###########################################################################################
""" Returns the hidden layer activations of the Autoencoder """


def feedForwardAutoencoder(theta, hidden_size, visible_size, input):
    """ Define limits to access useful data """

    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size

    """ Access W1 and b1 from 'theta' """

    W1 = theta[limit0: limit1].reshape(hidden_size, visible_size)
    b1 = theta[limit2: limit3].reshape(hidden_size, 1)

    """ Compute the hidden layer activations """

    hidden_layer = 1 / (1 + numpy.exp(-(numpy.dot(W1, input) + b1)))

    return hidden_layer


###########################################################################################
""" Loads data, trains the Autoencoder and Regressor, tests the accuracy """

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


def selfTaughtLearning():
    """ Define the parameters of the Autoencoder """

    vis_patch_side = 50  # side length of sampled image patches
    #hid_patch_side = 25  # side length of representative image patches ------------------to change here --------------------------------
    hid_patch_side = 16
    rho = 0.1  # desired average activation of hidden units
    lamda = 0.001  # weight decay parameter
    beta = 3  # weight of sparsity penalty term
    max_iterations = 800  # number of optimization iterations
    visible_size = vis_patch_side * vis_patch_side  # number of input units
    hidden_size = hid_patch_side * hid_patch_side  # number of hidden units

    """ Load MNIST images for training and testing """

    data,labels = dataset("gray50")

    opt_theta = theta()
    #opt_theta = opt_solution.x
    opt_W1 = opt_theta[0:16*16*2500].reshape(hidden_size, visible_size)



    print(len(opt_W1))
    """ Visualize the obtained optimal W1 weights """

    visualizeW1(opt_W1, vis_patch_side, hid_patch_side)
    numpy.savetxt('testfor16*16.txt', opt_theta, delimiter=',')

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

    train_features = feedForwardAutoencoder(opt_theta, hidden_size, visible_size, train_data)
    test_features = feedForwardAutoencoder(opt_theta, hidden_size, visible_size, test_data)

    #""" Initialize parameters of the Regressor """

    #input_size = 625  # input vector size                     #---------------------------to change here ---------------------#

    input_size = 256
    num_classes = 2  # number of classes
    lamda = 0.0001  # weight decay parameter
    max_iterations = 800  # number of optimization iterations

    regressor = SoftmaxRegression(input_size, num_classes, lamda)

    #""" Run the optimizer to get the optimal parameter values """

    opt_solution = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                           args=(train_features, train_labels,), method='L-BFGS-B',
                                           jac=True, options={'maxiter': max_iterations})
    opt_theta = opt_solution.x

    #""" Obtain predictions from the trained model """

    predictions = regressor.softmaxPredict(opt_theta, test_features)

    #""" Print accuracy of the trained model """

    correct = test_labels[:, 0] == predictions[:, 0]
    print("""Accuracy :""", numpy.mean(correct))


selfTaughtLearning()
