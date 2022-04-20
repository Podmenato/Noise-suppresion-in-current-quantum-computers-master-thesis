from SequentialPOVMMeasurement import SequentialPOVMMeasurement
import qiskit
from qiskit import *
from qiskit.quantum_info import random_unitary
from utilities import povm_bell
from qiskit.visualization import plot_histogram
from ProbabilisticMeasurement import ProbabilisticMeasurement


if __name__ == '__main__':
    qasm = qiskit.Aer.get_backend("qasm_simulator")

    seq = SequentialPOVMMeasurement(povm_bell, ["phi+", "phi-", "psi+", "psi-"])
    # Prepare measured state
    state = QuantumCircuit(2, 2)
    state.x(1)
    state.x(0)
    state.h(0)
    state.cnot(0, 1)

    prob = ProbabilisticMeasurement(povm_bell, ["phi+", "phi-", "psi+", "psi-"])
    results = prob.measure(state)
    prob.plot_histogram(results)