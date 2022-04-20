import numpy as np
from scipy.linalg import fractional_matrix_power
from qiskit.extensions import UnitaryGate
from qiskit import *
import matplotlib.pyplot as plt
from typing import List
from qiskit.providers.aer.noise.noise_model import NoiseModel
from qiskit.providers.aer.noise import ReadoutError


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

    # for i in range(qubits):
    #     appended.append(circuit.qubits[i])

    # SVD of B matrix into U, B diag and V
    u, b_diag, v = np.linalg.svd(b_measurement, full_matrices=True)
    # transform values into numpy arrays
    u = np.array(u)
    b_diag = np.array(b_diag)

    # create Ub and Ub dagger gates
    u_b_gate = UnitaryGate(u, label="U")
    u_b_dagger_gate = UnitaryGate(u.conj().transpose(), label="U+")

    # print(f"U = {u}")

    circuit.append(u_b_dagger_gate, circuit.qubits[0:qubits])

    for i in range(len(b_diag)):
        circuit.barrier()
        # make control gates
        # print(f"v[0]={calc_v(b_diag[0])}")
        vj = UnitaryGate(calc_v(b_diag[i])).control(qubits)

        x = (2 ** qubits) / 2

        for j in range(qubits):
            x_j = (2 ** j) / 2
            if j + 1 == qubits:
                if i < x:
                    circuit.x(circuit.qubits[j])
            else:
                if i % x < x_j:
                    circuit.x(circuit.qubits[j])

        circuit.append(vj, circuit.qubits[0:qubits + 1])

        for j in range(qubits):
            x_j = (2 ** j) / 2
            if j + 1 == qubits:
                if i < x:
                    circuit.x(circuit.qubits[j])
            else:
                if i % x < x_j:
                    circuit.x(circuit.qubits[j])

    circuit.barrier()

    circuit.append(u_b_gate, circuit.qubits[0:qubits])

    if measure:
        circuit.measure(circuit.qubits[qubits], measuring_clbit)

    circuit.reset(qubits)

    return circuit


def get_rotation_gate(base_vectors: List[np.array]):
    # Computation basis vectors generation
    computational_vectors = []
    for i in range(len(base_vectors)):
        vector = [0 for _el in base_vectors]
        vector[i] = 1
        computational_vectors.append(vector)

    matrices = [np.outer(computational_vectors[i], base_vectors[i]) for i in range(len(computational_vectors))]

    return UnitaryGate(np.sum(matrices, 0))


def plot_results_histogram(results: List[float], labels: List[str], title=None) -> None:
    keys = labels

    values = results

    font = {'size': 14}

    plt.figure(figsize=(3 * len(results) + 2, 10))
    plt.xlabel("Measured qubits", font)
    plt.ylabel("Counts", font)
    plt.bar(keys, values)
    xlocs, xlabs = plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v), font)
    plt.margins(0.2)
    plt.show()


## Scales the noise model.
##  - qscale is the common scale parameter for the quantum errors
##  - roscale is the common parameter for the readout errrors; if not
##    provided, the value of qscale is used
## The scale parameters should be from interval [0, 1]
## with 0 for ideal device and 1 for simulated device
def scale_noise(noise_model, qscale, roscale=None):
    if roscale is None:
        roscale = qscale

    ## Scaled version of the noise_model
    scaled = NoiseModel(basis_gates=noise_model.basis_gates)

    ## Quantum errors are stored in _local_quantum_errors attribute
    ## dictionary; _local_quantum_errors[operation][qubits] stores
    ## the Qiskit QuantumError in which probabilities are stored
    ## in _probs attribute
    qerrors = noise_model._local_quantum_errors
    for op in qerrors:
        for qubits in qerrors[op]:
            ## Copy the error
            newerror = qerrors[op][qubits].copy()

            ## Scale the probabilities
            probs = qscale * np.array(newerror.probabilities)
            probs[0] += 1 - qscale
            newerror._probs = probs

            ## Add error to the scaled noise model
            scaled.add_quantum_error(newerror, op, qubits)

    ## Readout errors are stored in _local_readout_errors attribute
    ## dictionary; _local_readout_errors[qubits] stores the Qiskit
    ## ReadoutError; we could change the probabilities as in the previous
    ## case, but here it is simpler to construct completely
    ## new ReadoutError
    roerrors = noise_model._local_readout_errors
    for qubits in roerrors:
        ## Reading probabilities
        probs = roerrors[qubits].probabilities

        ## Scale the probabilites
        error0 = roscale * probs[0][1]
        error1 = roscale * probs[1][0]

        ## Construction and addition of the error to the scaled noise model
        newerror = ReadoutError(np.array([[1 - error0, error0], [error1, 1 - error1]]))
        scaled.add_readout_error(newerror, qubits)

    return scaled

def map_number_to_qubit_result(number: int, qubits_count: int):
    binary_string = bin(number)
    qubit_result = binary_string[2:len(binary_string)]
    if len(qubit_result) != qubits_count:
        qubit_result = str(0*abs(len(qubit_result)-qubits_count)) + qubit_result
    return qubit_result

# Useful Matrices

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

__bell_phi_plus = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
__bell_phi_minus = np.array([[1 / 2, 0, 0, -1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [-1 / 2, 0, 0, 1 / 2]])
__bell_psi_plus = np.array([[0, 0, 0, 0], [0, 1 / 2, 1 / 2, 0], [0, 1 / 2, 1 / 2, 0], [0, 0, 0, 0]])
__bell_psi_minus = np.array([[0, 0, 0, 0], [0, 1 / 2, -1 / 2, 0], [0, -1 / 2, 1 / 2, 0], [0, 0, 0, 0]])
povm_bell = [__bell_phi_plus, __bell_phi_minus, __bell_psi_plus, __bell_psi_minus]
