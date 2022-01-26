import array

import numpy as np
import qiskit
from qiskit.visualization.array import array_to_latex
from POVM import *

from POVMMeasure import POVMMeasure
from qiskit.visualization import plot_histogram
from qiskit import *

from SequentialPOVMMeasurement import SequentialPOVMMeasurement

if __name__ == '__main__':
    povm_basic = [
        np.array([[1 / 6, 1 / 6],
                  [1 / 6, 1 / 6]]),
        np.array([[1 / 6, -1 / 6],
                  [-1 / 6, 1 / 6]]),
        np.array([[1 / 6, 0 + (-1j / 6)],
                  [0 + (1j / 6), 1 / 6]]),
        np.array([[1 / 6, 0 + (1j / 6)],
                  [0 + (-1j / 6), 1 / 6]]),
        np.array([[1 / 3, 0],
                  [0, 0]]),
        np.array([[0, 0],
                  [0, 1 / 3]])
    ]

    q = 1/4
    k = (1 / np.sqrt(3))*q
    m1 = np.sum([np.array([[q, 0], [0, q]]), np.array([[k, k * (1 - 1j)], [k * (1 + 1j), k * (-1)]])], 0)
    m2 = np.sum([np.array([[q, 0], [0, q]]), np.array([[k * (-1), k * (1 + 1j)], [k * (1 - 1j), k]])], 0)
    m3 = np.sum([np.array([[q, 0], [0, q]]), np.array([[k * (-1), k * (-1 - 1j)], [k * (-1 + 1j), k]])], 0)
    m4 = np.sum([np.array([[q, 0], [0, q]]), np.array([[k, k * (-1 + 1j)], [k * (-1 - 1j), k * (-1)]])], 0)
    povm_tetrahedron = [m1, m2, m3, m4]
    povm = POVM(povm_tetrahedron)
    valid = povm.validation()
    print(valid)

    # seq = SequentialPOVMMeasurement(povm_basic, ["x+", "x-", "y+", "y-", "z+", "z-"])
    # state = QuantumCircuit(1, 1)
    # state.h(0)
    # circuits = seq.make_circuits([["x+", "z-"], [["z+", "x-"], ["y+", "y-"]]], state)
    #
    # qasm = qiskit.Aer.get_backend("qasm_simulator")
    # circuits[0].q_circuit.draw("mpl")
    # job_1 = qiskit.execute(circuits[0].q_circuit, qasm, shots=1000)
    #
    # circuits[0].plot_histogram(job_1.result().get_counts())
    # plot_histogram(job_1.result().get_counts())
    # print(job_1.result().get_counts())
