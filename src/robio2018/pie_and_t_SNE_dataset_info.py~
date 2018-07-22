import plot_with_t_SNE
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams['font.size']=14.0

labels = 'TC','HC','HCO','NO','OS','WC','FP' # in counter-clockwise order
sizes = [17,18,18,14,18,17,6]
print 'total:%s'%sum(sizes)
explode = (0, 0.05, 0, 0, 0, 0,0 ) # only explode the 2nd slice
colors = ['b','g','r','c','m','y','gray']
fig, ax = plt.subplots()
fig.patch.set_facecolor('white') 
ax.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')

X = pd.read_csv('X_EXP.csv')
try:
    X = X.drop(['id'],axis=1)
except:
    pass
y = pd.read_csv('Y_EXP.csv', header=None,index_col=None, squeeze=True,usecols=[1])
X = X.values
y = y.values
plot_with_t_SNE.run(X,y)

plt.savefig('dataset_info.eps',format='eps',dpi=1000)
plt.show()
