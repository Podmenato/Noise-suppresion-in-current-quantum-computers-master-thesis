import copy
from typing import List, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from scipy.linalg import fractional_matrix_power


class SequentialPOVMMeasurementTree:
    def __init__(self, elements: List[np.array], labels: List[int], partitioning: list) -> None:
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
        self.labels = labels
        self.elements = elements
        self.dimension = elements[0].ndim

    def make_circuits(self, partitioning: list, circuit: QuantumCircuit, q_reg: QuantumRegister,
                      c_reg: ClassicalRegister):
        seq = SequentialPOVMMeasurementTree(self.elements, self.labels, partitioning)
        circuits = []
        SequentialPOVMMeasurement.__make_circuits_accum(seq, circuit, q_reg, c_reg, circuits)
        return circuits

    @staticmethod
    def __make_circuits_accum(seq: SequentialPOVMMeasurementTree,  circuit: QuantumCircuit, q_reg: QuantumRegister,
                              c_reg: ClassicalRegister, accumulator: List[QuantumCircuit]):
        b = np.sum(seq.result_measured, 0)
        print(seq.label)
        luder = SequentialPOVMMeasurement.luder_measurement(b, circuit, q_reg, c_reg)
        circuit += luder

        final_circuit = False
        if (seq.partitioning_measured is not None):
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(seq.partitioning_measured, circuit_copy, q_reg, c_reg, accumulator)
        else:
            accumulator.append(copy.deepcopy(circuit))

        if (seq.partitioning_other is not None):
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(seq.partitioning_other, circuit_copy, q_reg, c_reg, accumulator)

    @staticmethod
    def luder_measurement(b_measurement: np.array, circuit: QuantumCircuit, q_reg: QuantumRegister,
                          c_reg: ClassicalRegister):
        # SVD of B matrix into U, B diag and V
        u, b_diag, v = np.linalg.svd(b_measurement, full_matrices=True)
        # transform values into numpy arrays
        u = np.array(u)
        b_diag = np.array(b_diag)

        # create Ub and Ub* gates
        u_b_gate = UnitaryGate(u)
        u_b_dagger_gate = UnitaryGate(u.conj().transpose())

        circuit.append(u_b_gate, [q_reg[0]])
        circuit.append(u_b_gate, [q_reg[1]])

        # make control gates
        for i in range(0, 2):
            # vj = add_control(UnitaryGate(self.calc_v(i)), 1)
            vj = UnitaryGate(SequentialPOVMMeasurement.calc_v(b_diag[i])).control(1)
            circuit.append(vj, [q_reg[i], q_reg[2]])

        circuit.append(u_b_dagger_gate, [q_reg[0]])
        circuit.append(u_b_dagger_gate, [q_reg[1]])

        circuit.measure(q_reg[2], c_reg[2])

        return circuit

    @staticmethod
    def calc_v(l):
        return np.array([[(1 - l) ** (1 / 2), -(l ** (1 / 2))], [l ** (1 / 2), (1 - l) ** (1 / 2)]])

