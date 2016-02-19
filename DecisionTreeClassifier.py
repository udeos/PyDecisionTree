import pydot
import itertools
import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator


class BTree(object):
    def __init__(self, predicate, left, right, params):
        self.predicate = predicate
        self.left, self.right = left, right
        self.params = params

    def __str__(self):
        descr = []
        if self.predicate:
            descr.append('X[%s] <= %s' % (self.predicate[0], round(self.predicate[1], 4)))
        descr.extend(['%s = %s' % (k, v) for k, v in self.params.items()])
        return '\n'.join(descr)

    def to_graphviz(self):
        graph = pydot.Dot(graph_type='digraph')
        node_id = itertools.count()

        def fetch_nodes(parent, tree):
            node = pydot.Node(node_id.next(), label=str(tree))
            graph.add_node(node)
            if parent:
                graph.add_edge(pydot.Edge(parent, node))
            if tree.predicate:
                fetch_nodes(node, tree.left)
                fetch_nodes(node, tree.right)

        fetch_nodes(None, self)
        return graph


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.tree = None

    _entropy_vec = np.vectorize(lambda i, s: (i / float(s)) ** 2,
                                otypes=[np.float])

    def get_entropy(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        return 1 - self._entropy_vec(counts, float(Y.shape[0])).sum()

    def fit(self, X, Y):

        def build(XX, YY):
            params = {'entropy': self.get_entropy(YY)}
            left = right = None
            predicate = self.find_predicate(XX, YY)
            if predicate:
                f_id, f_val = predicate
                mask = XX[:, f_id] <= f_val
                left = build(XX[mask], YY[mask])
                right = build(XX[~mask], YY[~mask])
            else:
                params['class'] = YY[0]
            return BTree(predicate, left, right, params)

        self.tree = build(X, Y)

    def find_predicate(self, X, Y):
        min_entropy = self.get_entropy(Y)
        if min_entropy == 0.:
            return None
        predicate = None
        n_Y = len(Y)
        for obj in X:
            for f_id, f_val in enumerate(obj):
                mask = X[:, f_id] <= f_val
                y1, y2 = Y[mask], Y[~mask]
                n_y1, n_y2 = len(y1), len(y2)
                if n_y2 == 0 or n_y2 == 0:
                    continue
                e1 = self.get_entropy(y1) * n_y1
                e2 = self.get_entropy(y2) * n_y2
                local_entropy = (e1 + e2) / n_Y
                if local_entropy < min_entropy:
                    min_entropy = local_entropy
                    predicate = f_id, f_val
        return predicate

    def predict(self, X):

        def go_deep(tree, obj):
            if tree.predicate is None:
                return tree.params['class']
            f_id, f_val = tree.predicate
            if obj[f_id] <= f_val:
                return go_deep(tree.left, obj)
            return go_deep(tree.right, obj)

        leafs = np.array([go_deep(self.tree, obj) for obj in X])
        return leafs

    def draw_tree(self, filename):
        if self.tree is None:
            raise Exception("Tree is not built yet")
        self.tree.to_graphviz().write_png(filename)
