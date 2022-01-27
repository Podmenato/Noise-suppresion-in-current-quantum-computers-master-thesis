import numpy as np
from scipy.linalg import fractional_matrix_power
from qiskit.extensions import UnitaryGate
from qiskit import *


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

    # print(f"U = {u}")

    circuit.append(u_b_dagger_gate, [circuit.qubits[0]])

    circuit.barrier()
    # make control gates
    # print(f"v[0]={calc_v(b_diag[0])}")
    vj = UnitaryGate(calc_v(b_diag[0])).control(1)
    circuit.x(circuit.qubits[0])
    circuit.append(vj, [circuit.qubits[0], circuit.qubits[qubits]])
    circuit.x(circuit.qubits[0])

    circuit.barrier()
    # print(f"v[1]={calc_v(b_diag[1])}")
    vj = UnitaryGate(calc_v(b_diag[1])).control(1)
    circuit.append(vj, [circuit.qubits[0], circuit.qubits[qubits]])

    circuit.barrier()
    circuit.append(u_b_gate, [circuit.qubits[0]])

    if measure:
        circuit.measure(circuit.qubits[qubits], measuring_clbit)

    circuit.reset(qubits)

    return circuit

simple_povm_xyz = [
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

__q = 1 / 4
__k = (1 / np.sqrt(3)) * __q
__m1 = np.sum([np.array([[__q, 0], [0, __q]]), np.array([[__k, __k * (1 - 1j)], [__k * (1 + 1j), __k * (-1)]])], 0)
__m2 = np.sum([np.array([[__q, 0], [0, __q]]), np.array([[__k * (-1), __k * (1 + 1j)], [__k * (1 - 1j), __k]])], 0)
__m3 = np.sum([np.array([[__q, 0], [0, __q]]), np.array([[__k * (-1), __k * (-1 - 1j)], [__k * (-1 + 1j), __k]])], 0)
__m4 = np.sum([np.array([[__q, 0], [0, __q]]), np.array([[__k, __k * (-1 + 1j)], [__k * (-1 - 1j), __k * (-1)]])], 0)
povm_tetrahedron = [__m1, __m2, __m3, __m4]
