from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control


class POVMMeasure:
    """
    Wrapper class, will offer methods for different POVM implementations, and validates the POVM, unused right now
    """
    def __init__(self, elements: List[np.array]) -> None:
        self.elements = elements
        self.dimension = elements[0].ndim

    def validation(self) -> bool:
        """
        Checks if the POVM is valid, i.e. if its elements sum to identity and if the elements are positive semi definite
        :return: True if POVM is valid, False else
        """
        for element in self.elements:
            if not self.__is_positive_semi_definite(element):
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

    def sequential_measure(self, state: QuantumCircuit):
        # TODO
        return

    def statistical_measure(self):
        # TODO
        return

    def naimark_dilation_measure(self):
        # TODO
        return
