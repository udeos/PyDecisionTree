from sklearn import datasets, cross_validation
from DecisionTreeClassifier import DecisionTreeClassifier


X, Y = datasets.make_classification(n_samples=1000, n_features=10)
clf = DecisionTreeClassifier()
clf.fit(X, Y)
print cross_validation.cross_val_score(clf, X, Y)
