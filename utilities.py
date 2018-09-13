"""These are utilities to help with the scoring of a model."""

import numpy as np


def confusion_matrix(predictions, truths, classes):
    """Computes the confusion matrix of the predictions vs truths

    :param predictions: model predictions
    :param truths: true labels
    :param classes: an ordered set for the label possibilities
    :returns: a numpy array for the confusion matrix
    """

    matrix = np.zeros((len(classes), len(classes)), dtype=np.float32)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i, j] = [p == c1 and t == c2 for p,
                            t in zip(predictions, truths)].count(True)
    return matrix


"""This class is borrowed wholesale from the hagelslag repository as of 11 June 2018. It is reproduced here
with permission. Party responsible for copying: G. Eli Jergensen

@author David John Gagne <djgagne@ou.edu>"""


class MulticlassContingencyTable(object):
    """
    This class is a container for a contingency table containing more than 2 classes.
    The contingency table is stored in table as a numpy array with the rows corresponding to forecast categories,
    and the columns corresponding to observation categories.
    """

    def __init__(self, table=None, n_classes=2, class_names=("1", "0")):
        self.table = table
        self.n_classes = n_classes
        self.class_names = class_names
        if table is None:
            self.table = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def __add__(self, other):
        assert self.n_classes == other.n_classes, "Number of classes does not match"
        return MulticlassContingencyTable(self.table + other.table,
                                          n_classes=self.n_classes,
                                          class_names=self.class_names)

    def peirce_skill_score(self):
        """
        Multiclass Peirce Skill Score (also Hanssen and Kuipers score, True Skill Score)
        """
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (no * no).sum() / n ** 2)

    def gerrity_score(self):
        """
        Gerrity Score, which weights each cell in the contingency table by its observed relative frequency.
        :return:
        """
        k = self.table.shape[0]
        n = float(self.table.sum())
        p_o = self.table.sum(axis=0) / n
        p_sum = np.cumsum(p_o)[:-1]
        a = (1.0 - p_sum) / p_sum
        s = np.zeros(self.table.shape, dtype=float)
        for (i, j) in np.ndindex(*s.shape):
            if i == j:
                s[i, j] = 1.0 / (k - 1.0) * \
                    (np.sum(1.0 / a[0:j]) + np.sum(a[j:k-1]))
            elif i < j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:i]
                                                    ) - (j - i) + np.sum(a[j:k-1]))
            else:
                s[i, j] = s[j, i]
        return np.sum(self.table / float(self.table.sum()) * s)

    def heidke_skill_score(self):
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (nf * no).sum() / n ** 2)
