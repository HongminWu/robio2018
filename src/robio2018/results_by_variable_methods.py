import feature_extraction
import coloredlogs, logging, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn import decomposition
import matplotlib.pyplot as plt
import ipdb

coloredlogs.install()
logger = logging.getLogger()
logger.warning(os.path.basename(__file__))

def _metrics(y_test=None, y_pred=None):
    print (classification_report(y_test,y_pred))
    print ('accuracy: %s'% accuracy_score(y_test, y_pred))
    prfs = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    prf = filter(None, prfs)
    return prf

def _decisionTreeClassifier(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    logger.warning('DecisionTree(classification report:)')
    y_pred = cl.predict(X_test)
    return _metrics(y_test, y_pred)

def _naiveBayesClassifier(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    cl = GaussianNB()
    cl.fit(X_train, y_train)
    logger.warning('NaiveBayes(classification report:)')
    y_pred = cl.predict(X_test)
    return _metrics(y_test, y_pred)

def _linearSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 1.0, kernel = 'linear')
    cl.fit(X_train, y_train)
    logger.warning('LinearSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return _metrics(y_test, y_pred)

def _polynomialKernelSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 1.0, kernel = 'poly', gamma=1e-5)
    cl.fit(X_train, y_train)
    logger.warning('PolynomialKernelSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return _metrics(y_test, y_pred)
    
def _gaussianKernelSVMClassifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    cl = svm.SVC(C = 5.0, kernel = 'rbf')
    cl.fit(X_train, y_train)
    logger.warning('GaussianKernelSVM(classification report:)')
    y_pred = cl.predict(X_test)
    return _metrics(y_test, y_pred)


def _knnDTWClassifier(X_train, X_test, y_train, y_test):
    from K_Nearest_Neighbors_with_Dynamic_Time_Warping.util import KnnDtw
    cl = KnnDtw(n_neighbors = 1, max_warping_window=10)
    cl.fit(X_train.values, y_train.values)
    logger.warning('KNN+DTW@X(classification report:)')
    y_pred = cl.predict(X_test.values)
    return _metrics(y_test, y_pred[0])

def _pca(X_train, X_test, n_components=None):
    logger.warning('n_components:%s'%n_components)
    pca = decomposition.PCA()
    pca.fit(X_train)
    X_train_pca = np.dot(X_train, pca.components_[:, 0:n_components])
    X_test_pca  = np.dot(X_test, pca.components_[:, 0:n_components])
    return X_train_pca, X_test_pca

def _feature_selection(X_train, y_train, X_test, n_components=None):
    from sklearn.feature_selection import SelectKBest, f_classif
    logger.warning('n_components:%s'%n_components)    
    selector = SelectKBest(f_classif, k=n_components).fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    '''
    scores = -np.log10(selector.pvalues_)
    plt.figure()
    plt.bar(np.arange(X_train.shape[-1]), scores, width=.5,label='-$-log(p_{value})$', color = 'darkorange')
    plt.xlabel('Number of features')
    plt.savefig('scores.eps', format='eps', dpi=1000)
    plt.show()
    '''
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel
    
def run(X=None, X_filtered=None, y=None):
    
    if X is None or X_filtered is None or y is None:
        X, X_filtered, y = feature_extraction.run()

    X_train, X_test, X_filtered_train, \
        X_filtered_test, y_train, y_test = train_test_split(X,
                                                            X_filtered,
                                                                y,
                                                                test_size=.4,
                                                                stratify = y, random_state=0)

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
    
    # analyse with variable methods
    train_data = [X_train, X_filtered_train] 
    test_data  = [X_test, X_filtered_test]
    data_type  = ['X_train', 'X_filtered_train']

#----------------------------------------------------------------------------------------------------------------------------------------
 
    # precision_recall_fscore by different methods
    for train, test, d_type in zip(train_data, test_data, data_type):
        pre_rec_fc  = []
        logger.warning(d_type)
        pre_rec_fc.append(
            _decisionTreeClassifier(train, test, y_train, y_test))
        pre_rec_fc.append(
            _naiveBayesClassifier(train, test, y_train, y_test))
        pre_rec_fc.append(
            _linearSVMClassifier(train, test, y_train, y_test))
        pre_rec_fc.append(
            _polynomialKernelSVMClassifier(train, test, y_train, y_test))
        pre_rec_fc.append( _knnDTWClassifier(train, test, y_train, y_test))
            
        func_name = [ _decisionTreeClassifier.__name__,
                          _naiveBayesClassifier.__name__,
                          _linearSVMClassifier.__name__,
                          _polynomialKernelSVMClassifier.__name__,
                          _knnDTWClassifier.__name__,
                    ]
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        xs = np.arange(1, len(func_name)+1)
        pre_rec_fc = np.array(pre_rec_fc) 
        plt.bar(xs - .3, pre_rec_fc[:,0], width=.2, label='Precision', color='darkorange')
        plt.bar(xs - .1, pre_rec_fc[:,1], width=.2, label='Recall', color='navy')
        plt.bar(xs + .1, pre_rec_fc[:,2], width=.2, label='Fscore', color='c')
        fontsize = 16
        plt.ylabel('Precision-Recall-Fscore (%)', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)        
        plt.xlabel('Classifier', fontsize=fontsize)
        plt.xticks(xs, func_name,fontsize=fontsize)
        plt.legend(loc='upper right')
        plt.savefig('pre_rec_fc_by_classifiers.eps', format='eps', dpi=1000)
        plt.show()
            
    '''
    # trade-off the number of features
    for train, test, d_type in zip(train_data, test_data, data_type): 
        pre_rec_fc = []
        logger.warning(d_type)
        for n_components in np.arange(10, train.shape[1]-1, 50):
            ptrain, ptest = _feature_selection(train, y_train, test, n_components=n_components)
            acc =  _linearSVMClassifier(ptrain, ptest, y_train, y_test)
            pre_rec_fc.append(acc)
    #       acc = _decisionTreeClassifier(ptrain, ptest, y_train, y_test)
    #       acc =  _naiveBayesClassifier(ptrain, ptest, y_train, y_test)
    #       acc =  _linearSVMClassifier(ptrain, ptest, y_train, y_test)
    #       acc =  _polynomialKernelSVMClassifier(ptrain, ptest, y_train, y_test)
    #       acc =  _knnDTWClassifier(ptrain, ptest, y_train, y_test)

        plt.figure()
        plt.plot(np.arange(10, train.shape[1]-1, 50), pre_rec_fc)
        plt.grid()
        plt.ylabel('Pre_rec_fc (%)')
        plt.xlabel('Number of features')
        plt.savefig('acc_vs_features_%s.eps'%d_type, format='eps', dpi=1000)
        plt.show()
    '''
    
if __name__ == '__main__':
    X = pd.read_csv('X.csv')   
    X_filtered = pd.read_csv('X_filtered.csv')
    y = pd.read_csv('y.csv', header=None, index_col=None, squeeze=True, usecols=[1])
    run(X=X, X_filtered=X_filtered, y=y)
