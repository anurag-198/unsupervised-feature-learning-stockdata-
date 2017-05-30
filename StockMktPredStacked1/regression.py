import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from utils import dataset

def performRegression(train, test, train_output, test_output):
    """
    performs regression on returns using serveral algorithms
    """


    print ('Accuracy RFC: ', performRFReg(train, test, train_output, test_output))

    print ('Accuracy SVM: ', performSVMReg(train, test, train_output, test_output))

    print ('Accuracy BAG: ', performBaggingReg(train, test, train_output, test_output))

    print ('Accuracy ADA: ', performAdaBoostReg(train, test, train_output, test_output))

    print ('Accuracy BOO: ', performGradBoostReg(train, test, train_output, test_output))

    print('Accuracy KNN: ', performKNNReg(train, test, train_output, test_output))


def performRFReg(train, test, train_output, test_output):
    """
    Random Forest Regression
    """

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


def performSVMReg(train, test, train_output, test_output):
    """
    SVM Regression
    """

    clf = SVR()
    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


def performBaggingReg(train, test, train_output, test_output):
    """
    Bagging Regression
    """

    clf = BaggingRegressor()
    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


def performAdaBoostReg(train, test, train_output, test_output):
    """
    Ada Boost Regression
    """

    clf = AdaBoostRegressor()
    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


def performGradBoostReg(train, test, train_output, test_output):
    """
    Gradient Boosting Regression
    """

    clf = GradientBoostingRegressor()
    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


def performKNNReg(train, test, train_output, test_output):
    """
    KNN Regression
    """
    train = np.array(train)
    test = np.array(test)
    print(train.shape)
    print(test.shape)


    """for i in range(len(train)):
        if (len(train[i]) == 6):
            print("found")

    for i in range(len(test)):
        if (len(test[i]) == 6):
            print("found1")
    """

    clf = KNeighborsRegressor()
    clf.fit(train, train_output)

    Predicted = clf.predict(test)

    plt.plot(test_output)
    plt.plot(Predicted, color='red')
    plt.show()

    return np.sqrt(mean_squared_error(test_output, Predicted)), r2_score(test_output, Predicted)


#dat = dataset("CSV_Files_StockPrice")
#print(dat.shape)
#performRegression(dat, 0.8)