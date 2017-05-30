import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from utils import dataset

def performRegression(train, test, train_output, test_output):
    """
    performs regression on returns using serveral algorithms
    """

    #print ('Accuracy RFC: ', RFReg(train, test, train_output, test_output))

    print ('Accuracy SVM: ', SVMReg(train, test, train_output, test_output))

    #print ('Accuracy BAG: ', BaggingReg(train, test, train_output, test_output))

    #print('Accuracy KNN: ', KNNReg(train, test, train_output, test_output))


def RFReg(train, test, train_output, test_output):
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


def SVMReg(train, test, train_output, test_output):
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


def BaggingReg(train, test, train_output, test_output):
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


def KNNReg(train, test, train_output, test_output):
    """
    KNN Regression
    """
    train = np.array(train)
    test = np.array(test)
    print(train.shape)
    print(test.shape)

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