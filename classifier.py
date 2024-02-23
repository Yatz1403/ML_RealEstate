from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
features = iris.data
labels  = iris.target
print(features[0], labels[0])

clf= KNeighborsClassifier()
clf.fit(features,labels)
pred = clf.predict([[1,1,1,1]])
print(pred)
