import numpy as np
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
        np.array([[1/3, 0],
                  [0, 0]]),
        np.array([[0, 0],
                  [0, 1/3]])
    ]

    seq = SequentialPOVMMeasurement(povm_basic, ["x+", "x-", "y+", "y-", "z+", "z-"])
    state = QuantumCircuit(1, 1)
    state.h(0)
    circuits = seq.make_circuits([["x+", "z-"], [["z+", "x-"], ["y+", "y-"]]], state)

    qasm = qiskit.Aer.get_backend("qasm_simulator")
    circuits[0].q_circuit.draw("mpl")
    job_1 = qiskit.execute(circuits[0].q_circuit, qasm, shots=1000)

    circuits[0].plot_histogram(job_1.result().get_counts())
    plot_histogram(job_1.result().get_counts())
    print(job_1.result().get_counts())

