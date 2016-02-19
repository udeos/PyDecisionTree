from sklearn import datasets, cross_validation

from DecisionTreeClassifier import DecisionTreeClassifier


iris = datasets.load_iris()
X, Y = iris.data, iris.target
clf = DecisionTreeClassifier()
clf.fit(X, Y)
print cross_validation.cross_val_score(clf, X, Y)
clf.draw_tree('DecisionTreeClassifier_visual.png')
