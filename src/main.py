import numpy as np
from POVMMeasure import POVMMeasure
from qiskit import *

from SequentialPOVMMeasurement import SequentialPOVMMeasurement, SequentialPOVMMeasurementTree

if __name__ == '__main__':
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(3, 'c')
    circ = QuantumCircuit(q, c)

    povm_simple = POVMMeasure([np.array([[0.5, 0], [0, 0.5]]), np.array([[0.5, 0], [0, 0.5]])])

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
    circuits = seq.make_circuits([[1, 2], [[3, 4], [5, 6]]], circ, q, c)
    print(circuits)
