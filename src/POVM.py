import numpy as np
from typing import List


class Effect:
    def __init__(self, matrix: np.array, label):
        """
        Represents an effect, contains matrix and label corresponding to the effect
        :param matrix: Matrix of the effect
        :param label: Label corresponding to the effect
        """
        self.matrix = matrix
        self.label = label


class POVM:
    def __init__(self, effects: List[np.array], labels=None):
        """
        Represents POVM, contains list of effects and methods for validation
        :param effects: Effects of the POVM
        :param labels: Labels corresponding to the effects
        """
        if labels is None:
            labels = []
        if len(labels) == 0:
            for i in range(len(effects)):
                labels.append(str(i))

        if len(effects) != len(labels):
            raise ValueError("The labels dont match the effects")

        if len(set(labels)) != len(labels):
            raise ValueError("Labels are not unique")

        elements = []

        for i in range(len(effects)):
            effect = Effect(effects[i], labels[i])
            elements.append(effect)

        self.elements = elements
        self.dimension = elements[0].matrix.shape[0]

    def validation(self) -> bool:
        """
        Checks if the POVM is valid, i.e. if its elements sum to identity and if the elements are positive semi definite
        :return: True if POVM is valid, False else
        """
        for element in self.elements:
            if not self.__is_positive_semi_definite(element.matrix):
                return False
        return self.__sums_to_identity()

    @staticmethod
    def __is_positive_semi_definite(element: np.array, tolerance=+1e-7) -> bool:
        """
        Checks if the element is positive semi definite
        :param element: numpy array matrix element
        :return: True if element is positive semi definite, False else
        """
        return np.all(np.isclose(element, element.conj().transpose())) & np.all(np.linalg.eigvals(element) + tolerance >= 0)

    def __sums_to_identity(self) -> bool:
        """
        Checks if the POVM elements sum to identity
        :return: True if the POVM elements sum to identity, False else
        """
        element_matrices = []
        for e in self.elements:
            element_matrices.append(e.matrix)
        return np.all(np.isclose(np.sum(element_matrices, 0), np.identity(self.dimension)))
