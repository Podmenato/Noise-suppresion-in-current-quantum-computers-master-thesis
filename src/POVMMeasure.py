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



    def sequential_measure(self, state: QuantumCircuit):
        # TODO
        return

    def statistical_measure(self):
        # TODO
        return

    def naimark_dilation_measure(self):
        # TODO
        return
