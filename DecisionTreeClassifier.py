import pydot
import itertools
import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator


class BTree(object):
    label = None

    def __init__(self, proportions, predicate=None, left=None, right=None):
        self.predicate = predicate
        self.left, self.right = left, right
        self.proportions = proportions
        if predicate is None:
            self.label = max(proportions, key=proportions.get)

    def __str__(self):
        descr = []
        if self.predicate:
            descr.append('X[%s] <= %s' % (self.predicate[0], round(self.predicate[1], 4)))
        if self.label:
            descr.append('Lb = %s' % self.label)
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
        self.n_features = None

    def _is_valid(self):
        if self.tree is None:
            raise Exception("Tree is not built yet")

    @staticmethod
    def _get_proportions(Y):
        keys, counts = np.unique(Y, return_counts=True)
        return dict(zip(keys, counts / float(len(Y))))

    def _build(self, D):
        kwargs = {
            'proportions': self._get_proportions(D['label'].flatten()),
        }
        predicate = self._find_predicate(D)
        if predicate:
            f_id, f_val = predicate
            mask = D[str(f_id)] <= f_val
            kwargs['predicate'] = predicate
            kwargs['left'] = self._build(D[mask])
            kwargs['right'] = self._build(D[~mask])
        return BTree(**kwargs)

    def _find_predicate(self, D):
        No = float(D.shape[0])
        Nf = self.n_features
        uniques, counts = np.unique(D['label'], return_counts=True)
        min_impurity = 1 - sum([(i / No) ** 2 for i in counts])
        if min_impurity == 0.:
            return None
        predicate = None
        for fid in range(Nf):
            left = dict.fromkeys(uniques, 0)
            right = dict(zip(uniques, counts))
            Dp = np.sort(D[[str(fid), 'label']], order=str(fid), axis=0)
            for label, objs in itertools.groupby(Dp.flatten(), lambda x: x[1]):
                objs = list(objs)
                left[label] += len(objs)
                right[label] -= len(objs)
                left_values, right_values = left.values(), right.values()
                left_num, right_num = float(sum(left_values)), float(sum(right_values))
                if right_num == 0:
                    continue
                left_impurity = 1 - sum([(i / left_num) ** 2 for i in left_values])
                right_impurity = 1 - sum([(i / right_num) ** 2 for i in right_values])
                current_impurity = (left_impurity * left_num + right_impurity * right_num) / No
                if current_impurity < min_impurity:
                    min_impurity = current_impurity
                    predicate = fid, objs.pop()[0]
        return predicate

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        D = np.concatenate((X, Y.reshape(Y.shape[0], 1)), axis=1)
        dt = [(str(i), 'f8') for i in range(self.n_features)]
        dt.append(('label', 'f8'))
        D = D.view(dt)
        self.tree = self._build(D)

    def predict(self, X):

        def go_deep(tree, obj):
            if tree.predicate is None:
                return tree.label
            f_id, f_val = tree.predicate
            if obj[f_id] <= f_val:
                return go_deep(tree.left, obj)
            return go_deep(tree.right, obj)

        self._is_valid()
        return np.array([go_deep(self.tree, obj) for obj in X])

    def draw_tree(self, filename):
        self._is_valid()
        self.tree.to_graphviz().write_png(filename)
