import numpy as np
from typing import List


class Effect:
    def __init__(self, matrix:np.array, label):
        self.matrix = matrix
        self.label = label

class POVM:
    def __init__(self, effects: List[np.array], labels: List[int]):
        if len(effects) != len(labels):
            raise ValueError("The labels dont match the effects")

        elements = []

        for i in range(len(effects)):
            effect = Effect(effects[i], labels[i])
            elements.append(effect)

        self.elements = elements
        self.dimension = elements[0].matrix.ndim

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
    def __is_positive_semi_definite(element: np.array) -> bool:
        """
        Checks if the element is positive semi definite
        :param element: numpy array matrix element
        :return: True if element is positive semi definite, False else
        """
        return np.array_equal(element, element.conj().transpose()) & np.all(np.linalg.eigvals(element) >= 0)

    def __sums_to_identity(self) -> bool:
        """
        Checks if the POVM elements sum to identity
        :return: True if the POVM elements sum to identity, False else
        """
        return np.array_equal(np.sum(self.elements, 0), np.identity(self.dimension))
