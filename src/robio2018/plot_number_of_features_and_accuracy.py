from classifiers_and_metrics import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':16})
import numpy as np
import pandas as pd
import sys, ipdb

def plot_feature_scores(X, y):
    newX, scores = feature_selection(X, y, n_components=X.shape[-1])

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    
    plt.bar(np.arange(scores.shape[-1]), scores, width=.5, label='$-log(p_{value})$', color = 'darkorange')
    plt.xlabel('Feature Index')    
    plt.ylabel('Score')
    plt.title('Feature Importance Evaluation')
    plt.xlim(-5, 560)
    plt.ylim(0,47)
    plt.legend(loc=2)
    plt.savefig('scores.eps', format='eps', dpi=300)
    
    sorted_scores = np.sort(scores)[::-1]
    sorted_scores = sorted_scores[~np.isnan(sorted_scores)]
    nFeatures = np.arange(10, sorted_scores.shape[-1], 30)
    percentages = []
    for nFeature in nFeatures:
       percentages.append(sum(sorted_scores[:nFeature]) / sum(sorted_scores))
    percentages = np.array(percentages)*100
    segments = percentages.shape[-1]
    data = nFeatures - 10
    fig = plt.figure(figsize=(14,3))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)

    colors ='r'
    patch_handles = []
    left = np.zeros(1) # left alignment of data starts at zero
    for i, d in enumerate(data):
        patch_handles.append(ax.barh(1, d, color=colors[i%len(colors)],alpha=.5, align='center', left=left))
        left = d
    # go through all of the bar segments and annotate
    for j in xrange(len(patch_handles)):
        for i, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
#            x = 0.5*patch.get_width() + bl[0]
            x = nFeatures[j]
            y = 0.5*patch.get_height() + bl[1]
            ax.text(x,y, "%d%%" % (percentages[j]), ha='center')
    plt.xlim(0, nFeatures[-1])
    plt.title('The proportion of relevant features')
    ax.set_yticks([])
    ax.set_xticks(np.arange(10, nFeatures[-1], 50))    
    ax.set_ylabel('Percentage')
    ax.set_xlabel('The maximum $N$ scores')
    plt.savefig('percentages.eps', format='eps', dpi=1000)    
    plt.show()

if __name__ == '__main__':
    X = pd.read_csv('X.csv')
    try:
        X = X.drop(['id'],axis=1)
    except:
        pass
    y = pd.read_csv('y.csv', header=None, index_col=None, squeeze=True, usecols=[1])
    plot_feature_scores(X,y)
