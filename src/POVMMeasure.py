from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control


class POVMMeasure:
    def __init__(self, elements: List[np.array]) -> None:
        self.elements = elements
        self.dimension = elements[0].ndim

    def validation(self) -> bool:
        for element in self.elements:
            if not self.__is_positive_semi_definite(element):
                return False
        return self.__sums_to_identity()

    @staticmethod
    def __is_positive_semi_definite(element: np.array) -> bool:
        return np.array_equal(element, element.conj().transpose()) & np.all(np.linalg.eigvals(element) >= 0)

    def __sums_to_identity(self) -> bool:
        return np.array_equal(np.sum(self.elements, 0), np.identity(self.dimension))

    def sequential_measure(self, state: QuantumCircuit):
        return

    def statistical_measure(self):
        return

    def naimark_dilation_measure(self):
        return
