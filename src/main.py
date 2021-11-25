import numpy as np
from POVMMeasure import POVMMeasure
from qiskit import *

from SequentialPOVMMeasurement import SequentialPOVMMeasurement, SequentialPOVMMeasurementTree

if __name__ == '__main__':
    povm_effects = [
        np.array([[1 / 6, 1 / 6],
                  [1 / 6, 1 / 6]]),
        np.array([[1 / 6, -1 / 6],
                  [-1 / 6, 1 / 6]]),
        np.array([[1 / 6, 0 - (1j / 6)],
                  [0 + (1j / 6), 1 / 6]]),
        np.array([[1 / 6, 0 + (1j / 6)],
                  [0 - (1j / 6), 1 / 6]]),
        np.array([[1 / 3, 0],
                  [0, 0]]),
        np.array([[0, 0],
                  [0, 1 / 3]])
    ]

    seq = SequentialPOVMMeasurement(povm_effects, [1, 2, 3, 4, 5, 6])
    state = QuantumCircuit(1, 1)
    state.h(0)
    circuits = seq.make_circuits([[1, 4], [[2, 5], [3, 6]]], state)
    print(circuits[0][0])
    print(state.qubits)
