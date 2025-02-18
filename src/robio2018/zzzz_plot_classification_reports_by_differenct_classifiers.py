from classifiers_and_metrics import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ipdb

def run(X=None, y=None):
    train, test, y_train, y_test = train_test_split(X,y,test_size=.4, stratify = y, random_state=0)

    # precision_recall_fscore by different methods
    pre_rec_fsc_acc  = []
    pre_rec_fsc_acc.append(decisionTreeClassifier(train, test, y_train, y_test))
    pre_rec_fsc_acc.append(naiveBayesClassifier(train, test, y_train, y_test))
    pre_rec_fsc_acc.append(linearSVMClassifier(train, test, y_train, y_test))
    pre_rec_fsc_acc.append(polynomialKernelSVMClassifier(train, test, y_train, y_test))
    pre_rec_fsc_acc.append(knnDTWClassifier(train, test, y_train, y_test))

    func_name = [ 'DecisionTree','NaiveBayes','LinearSVM','PolynomialSVM', 'KNN+DTW']
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    xs = np.arange(len(func_name))
    pre_rec_fsc_acc = np.array(pre_rec_fsc_acc)
    w = 0.3
    fontsize = 12
    plt.bar(xs - w/2.0 - w, pre_rec_fsc_acc[:,0], width=w, label='Precision', color='darkorange')
    plt.bar(xs - w/2.0,     pre_rec_fsc_acc[:,1],     width=w, label='Recall', color='navy')
    plt.bar(xs + w/2.0,     pre_rec_fsc_acc[:,2],     width=w, label='Fscore', color='c')
    plt.ylabel('Precision-Recall-Fscore (%)', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)        
    plt.xlabel('Classifier', fontsize=fontsize)
    plt.xticks(xs, func_name,fontsize=fontsize-2)
    plt.legend(bbox_to_anchor = (0., 1.02, 1.0, 0.102), loc=3, ncol=3, mode='expand', borderaxespad=0.)
    plt.savefig('pre_rec_fsc_acc_by_classifiers.eps', format='eps', dpi=1000)
    plt.show()

if __name__ == '__main__':
    X = pd.read_csv('X.csv')
    try:
        X = X.drop(['id'],axis=1)
    except:
        pass
    y = pd.read_csv('y.csv', header=None, index_col=None, squeeze=True, usecols=[1])
    run(X=X, y=y)


    
