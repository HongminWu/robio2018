import feature_extraction
import coloredlogs, logging, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn import decomposition
import matplotlib.pyplot as plt
import ipdb

coloredlogs.install()
logger = logging.getLogger()
logger.warning(os.path.basename(__file__))

def metrics(y_test=None, y_pred=None):
    print (classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy: %s'% accuracy)
    prfs = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    prf  = filter(None, prfs)
    prfa  = prf + (accuracy,)
    return prfa

def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    logger.warning('DecisionTree(classification report:)')
    y_pred = cl.predict(X_test)
    return metrics(y_test, y_pred)

def naiveBayesClassifier(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    cl = GaussianNB()
    cl.fit(X_train, y_train)
    logger.warning('NaiveBayes(classification report:)')
    y_pred = cl.predict(X_test)
    return metrics(y_test, y_pred)

def linearSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 1.0, kernel = 'linear')
    cl.fit(X_train, y_train)
    logger.warning('LinearSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return metrics(y_test, y_pred)

def polynomialKernelSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 1.0, kernel = 'poly', gamma=1e-5)
    cl.fit(X_train, y_train)
    logger.warning('PolynomialKernelSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return metrics(y_test, y_pred)
    
def gaussianKernelSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 5.0, kernel = 'rbf')
    cl.fit(X_train, y_train)
    logger.warning('GaussianKernelSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return metrics(y_test, y_pred)


def knnDTWClassifier(X_train, X_test, y_train, y_test):
    from K_Nearest_Neighbors_with_Dynamic_Time_Warping.util import KnnDtw
    cl = KnnDtw(n_neighbors = 1, max_warping_window=10)
    cl.fit(X_train.values, y_train.values)
    logger.warning('KNN+DTW@X(classification report:)')
    y_pred = cl.predict(X_test.values)
    return metrics(y_test, y_pred[0])

def pca(X_train=None, y_train=None, X_test=None, n_components=None):
    logger.warning('n_components:%s'%n_components)
    pca = decomposition.PCA()
    ipdb.set_trace()
    pca.fit(X_train)
    X_train_pca = np.dot(X_train, pca.components_[:, 0:n_components])
    X_test_pca  = np.dot(X_test, pca.components_[:, 0:n_components])
    return X_train_pca, X_test_pca

def feature_selection(X, y, n_components=None):
    from sklearn.feature_selection import SelectKBest, f_classif
    logger.warning('n_components:%s'%n_components)    
    selector = SelectKBest(f_classif, k=n_components).fit(X, y)
    X = selector.transform(X)
    scores = -np.log10(selector.pvalues_)
    return X, scores
    
    '''
    # multiclass feature selection
    from tsfresh import select_features
    selected_features = set()
    for label in y.unique():
        y_train_binary = y_train == label
        X_train_selected = select_features(X_train, y_train_binary)
        logger.info('Number of selected features for class{}:{}/{}'.format(label,X_train_selected.shape[1], X_train.shape[1]))
        selected_features = selected_features.union(set(X_train_selected.columns))
    X_selected_train = X_train[list(selected_features)]
    X_selected_test  = X_test[list(selected_features)]
    logger.warning('X_selected_train:{}; X_selected_test:()'.format(X_selected_train.shape, X_selected_test.shape))
    '''
