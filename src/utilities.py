import numpy as np
from scipy.linalg import fractional_matrix_power
from qiskit.extensions import UnitaryGate
from qiskit import *
from qiskit.circuit.add_control import add_control
import matplotlib.pyplot as plt


def measurement_change(b: np.array, effect: np.array):
    b_pinv = np.linalg.pinv(fractional_matrix_power(b, 0.5))

    return np.matmul(np.matmul(b_pinv, effect), b_pinv)


def calc_v(eigenvalue):
    """
    Calculates V matrix from the eigenvalue on the B diagonal matrix
    :param eigenvalue: the value from B diagonal
    :return: V matrix
    """
    eigenvalue = np.round(eigenvalue, 14)
    return np.array([[(1 - eigenvalue) ** (1 / 2), -(eigenvalue ** (1 / 2))],
                     [eigenvalue ** (1 / 2), (1 - eigenvalue) ** (1 / 2)]])


def luder_measurement(b_measurement: np.array, qubits: int, cbits: int, measuring_clbit=0,
                      measure=True) -> QuantumCircuit:
    """
    Returns a circuit representing the Luder measurement
    :param b_measurement: Numpy array matrix, representing the coarse graining of POVM measurements
    :return: QuantumCircuit
    """
    # create circuit
    circuit = QuantumCircuit(qubits + 1, cbits)

    # SVD of B matrix into U, B diag and V
    u, b_diag, v = np.linalg.svd(b_measurement, full_matrices=True)
    # transform values into numpy arrays
    u = np.array(u)
    b_diag = np.array(b_diag)

    # create Ub and Ub dagger gates
    u_b_gate = UnitaryGate(u, label="U")
    u_b_dagger_gate = UnitaryGate(u.conj().transpose(), label="U+")

    print(f"U = {u}")

    circuit.append(u_b_dagger_gate, [circuit.qubits[0]])

    circuit.barrier()
    # make control gates
    print(f"v[0]={calc_v(b_diag[0])}")
    vj = UnitaryGate(calc_v(b_diag[0])).control(1)
    circuit.x(circuit.qubits[0])
    circuit.append(vj, [circuit.qubits[0], circuit.qubits[qubits]])
    circuit.x(circuit.qubits[0])

    circuit.barrier()
    print(f"v[1]={calc_v(b_diag[1])}")
    vj = UnitaryGate(calc_v(b_diag[1])).control(1)
    circuit.append(vj, [circuit.qubits[0], circuit.qubits[qubits]])

    circuit.barrier()
    circuit.append(u_b_gate, [circuit.qubits[0]])

    if measure:
        circuit.measure(circuit.qubits[qubits], measuring_clbit)

    circuit.reset(qubits)

    return circuit


def plot_povm_histogram(data, title=None):
    keys = list(data.keys())

    length = len(keys[0])
    names = ['1'*length, '0'+'1'*(length-1), 'rest']

    rest = 0
    for key in keys:
        if key not in names:
            rest += data[key]

    measured = 0
    if names[0] in keys:
        measured = data[names[0]]

    measured_opposite = 0
    if names[1] in keys:
        measured_opposite = data[names[1]]

    values = [measured/1000, measured_opposite/1000, rest/1000]

    font = {'size': 14}

    plt.figure()
    plt.xlabel("Measured qubits", font)
    plt.ylabel("Counts", font)
    plt.bar(names, values)
    xlocs, xlabs = plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v), font)
    plt.margins(0.2)
    plt.show()
