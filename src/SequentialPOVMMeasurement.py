import copy
from typing import List, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from scipy.linalg import fractional_matrix_power

class SequentialPOVMMeasurementTree:
    def __init__(self, elements: List[np.array], labels: List[int], partitioning: list) -> None:
        """
        Internal structure representing the sequential POVM measurement tree, based on the entered partitioning
        :param elements: list of all element matrices
        :param labels: TODO implement labels
        :param partitioning: list of indices, representing the partitioning, i.e. [[1-2],[3-4]]
        TODO instead of indices, use labels
        """
        i1, i2 = SequentialPOVMMeasurementTree.__get_results_idx_list(partitioning)

        effects1 = []
        effects2 = []
        for i in i1:
            effects1.append(elements[i - 1])

        for i in i2:
            effects2.append(elements[i - 1])
        self.result_measured = effects1
        self.result_other = effects2

        self.label = ''.join(map(str, i1)) + '-' + ''.join(map(str, i2))

        self.partitioning_measured = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects1, i1,
                                                                                           partitioning[0])

        self.partitioning_other = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects2, i2,
                                                                                        partitioning[1])

    @staticmethod
    def __get_results_idx_list(partitioning: list):
        if len(partitioning) != 2:
            raise Exception("Invalid partitioning format")

        if type(partitioning[0]) is not type(partitioning[1]):
            raise Exception("Invalid partitioning format")

        if isinstance(partitioning[0], int):
            return [partitioning[0]], [partitioning[1]]

        p1 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[0])
        p2 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[1])

        return p1, p2

    @staticmethod
    def __flatten_partitioning(partitioning: list):
        if len(partitioning) > 2 | len(partitioning) == 0:
            raise Exception("Invalid partitioning format")

        if len(partitioning) == 1:
            if isinstance(partitioning[0], int):
                return partitioning
            else:
                raise Exception("Invalid partitioning format")

        if type(partitioning[0]) is not type(partitioning[1]):
            raise Exception("Invalid partitioning format")

        if isinstance(partitioning[0], int):
            return partitioning
        elif isinstance(partitioning[0], list):
            p1 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[0])
            p2 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[1])
            return [*p1, *p2]

    @staticmethod
    def __measurement_change(b: np.array, effect: np.array):
        b_pinv = np.linalg.pinv(fractional_matrix_power(b, 0.5))

        return np.matmul(np.matmul(b_pinv, effect), b_pinv)

    @staticmethod
    def __make_sub_measurements(all_elements: List[np.array], measured_elements: List[np.array], indices: List[int],
                                partitioning: list):
        if len(measured_elements) != 1:
            new_elements = []
            for i in range(0, len(all_elements)):
                if i + 1 in indices:
                    b = np.sum(measured_elements, 0)
                    new_elements.append(SequentialPOVMMeasurementTree.__measurement_change(b, all_elements[i]))
                else:
                    new_elements.append(None)
            return SequentialPOVMMeasurementTree(new_elements, [], partitioning)
        return None


class SequentialPOVMMeasurement:
    def __init__(self, elements: List[np.array], labels: List[int]) -> None:
        """
        Class used to create circuits for POVM representation using the sequential measurement method
        :param elements: list of all element matrices
        :param labels: TODO
        """
        self.labels = labels
        self.elements = elements

    def make_circuits(self, partitioning: list, state: QuantumCircuit):
        """
        Creates circuits, each measures its specific measurement
        :param state: circuit with prepared state
        :param partitioning: list of indices, each corresponding to the index in the element in elements array.
            Should be in the following format [[1, 2],[3, 4]]. Note, [1,[2, 3]] is not valid, [[1], [2, 3]] is.
        :return: List of QuantumCircuits
        """
        # create measurement tree structure
        seq = SequentialPOVMMeasurementTree(self.elements, self.labels, partitioning)
        circuits = []
        qubits_count = len(state.qubits)
        base_circuit = QuantumCircuit(qubits_count + 1, qubits_count + 1)
        base_circuit = base_circuit.compose(state)
        SequentialPOVMMeasurement.__make_circuits_accum(self, seq, base_circuit, circuits)
        return circuits

    def __make_circuits_accum(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit,
                              accumulator: List[QuantumCircuit]):
        """
        Helper method used to traverse the SequentialPOVMMeasurementTree and create circuits from it
        :param seq: Internal tree structure
        :param circuit: quantum circuit on to which new addition should be appended
        :param accumulator: list of already finished circuits
        :return: None
        """

        # current measurement
        b = np.sum(seq.result_measured, 0)

        # create luder measurement circuit based on b
        luder = SequentialPOVMMeasurement.luder_measurement(self, b, len(circuit.qubits) - 1)
        circuit += luder

        # traversing deeper into the seq structure
        if seq.partitioning_measured is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_measured, circuit_copy, accumulator)
        else:
            accumulator.append(copy.deepcopy(circuit))

        if seq.partitioning_other is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_other, circuit_copy, accumulator)

    def make_single_circuit(self, partitioning: list):
        """
        WIP
        :param partitioning:
        :return:
        """
        # TODO implement method
        seq = SequentialPOVMMeasurementTree(self.elements, self.labels, partitioning)

    def __make_single_circuit_helper(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit):
        """
        WIP
        :param seq:
        :param circuit:
        :return:
        """
        # TODO implement method
        b = np.sum(seq.result_measured, 0)


    def luder_measurement(self, b_measurement: np.array, qubits: int) -> QuantumCircuit:
        """
        Returns a circuit representing the Luder measurement
        :param b_measurement: Numpy array matrix, representing the coarse graining of POVM measurements
        :return: QuantumCircuit
        """
        # create circuit
        circuit = QuantumCircuit(qubits + 1, qubits + 1)

        # SVD of B matrix into U, B diag and V
        u, b_diag, v = np.linalg.svd(b_measurement, full_matrices=True)
        # transform values into numpy arrays
        u = np.array(u)
        b_diag = np.array(b_diag)

        # create Ub and Ub dagger gates
        u_b_gate = UnitaryGate(u)
        u_b_dagger_gate = UnitaryGate(u.conj().transpose())

        for i in range(0, qubits):
            circuit.append(u_b_gate, [circuit.qubits[i]])

        # make control gates
        for i in range(0, qubits):
            vj = UnitaryGate(SequentialPOVMMeasurement.calc_v(b_diag[i])).control(1)
            circuit.append(vj, [circuit.qubits[i], circuit.qubits[qubits]])

        for i in range(0, qubits):
            circuit.append(u_b_gate, [circuit.qubits[i]])

        circuit.measure(circuit.qubits[qubits], circuit.clbits[qubits])

        return circuit

    @staticmethod
    def calc_v(eigenvalue):
        """
        Calculates V matrix from the eigenvalue on the B diagonal matrix
        :param eigenvalue: the value from B diagonal
        :return: V matrix
        """
        eigenvalue = np.round(eigenvalue, 14)
        return np.array([[(1 - eigenvalue) ** (1 / 2), -(eigenvalue ** (1 / 2))],
                         [eigenvalue ** (1 / 2), (1 - eigenvalue) ** (1 / 2)]])
