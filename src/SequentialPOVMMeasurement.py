import copy
from typing import List, Union, Tuple
import numpy as np
from qiskit import *
from utilities import measurement_change, luder_measurement


class SequentialPOVMMeasurementTree:
    def __init__(self, elements: List[np.array], labels: List[int], partitioning: list, depth=0, measuring_label="(0/1)") -> None:
        """
        Internal structure representing the sequential POVM measurement tree, based on the entered partitioning
        :param elements: list of all element matrices
        :param labels: TODO implement labels
        :param partitioning: list of indices, representing the partitioning, i.e. [[1-2],[3-4]]
        :param depth: depth of the node inside the tree
        TODO instead of indices, use labels
        """
        self.depth = depth + 1
        i1, i2 = SequentialPOVMMeasurementTree.__get_results_idx_list(partitioning)

        effects1 = []
        effects2 = []
        for i in i1:
            effects1.append(elements[i - 1])

        for i in i2:
            effects2.append(elements[i - 1])
        self.result_measured = effects1
        self.result_other = effects2

        self.label = ''.join(map(str, i1)) + '-' + ''.join(map(str, i2)) + ' ' + measuring_label

        self.partitioning_measured = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects1, i1,
                                                                                           partitioning[0], depth=self.depth, measuring_label='0'+measuring_label)

        self.partitioning_other = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects2, i2,
                                                                                        partitioning[1], depth=self.depth, measuring_label='1'+measuring_label)

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
    def __make_sub_measurements(all_elements: List[np.array], measured_elements: List[np.array], indices: List[int],
                                partitioning: list, depth: int, measuring_label: str):
        if len(measured_elements) != 1:
            new_elements = []
            for i in range(0, len(all_elements)):
                if i + 1 in indices:
                    b = np.sum(measured_elements, 0)
                    new_elements.append(measurement_change(b, all_elements[i]))
                else:
                    new_elements.append(None)
            return SequentialPOVMMeasurementTree(new_elements, [], partitioning, depth, measuring_label=measuring_label)
        return None

    def largest_depth(self):
        measured_depth = self.depth
        other_depth = self.depth

        if self.partitioning_measured is not None:
            measured_depth = self.partitioning_measured.largest_depth()

        if self.partitioning_other is not None:
            other_depth = self.partitioning_other.largest_depth()

        if measured_depth > other_depth:
            return measured_depth
        elif other_depth > measured_depth:
            return other_depth
        else:
            return measured_depth


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
        classical_count = seq.largest_depth()
        base_circuit = QuantumCircuit(qubits_count + 1, classical_count)
        base_circuit = base_circuit.compose(state)
        SequentialPOVMMeasurement.__make_circuits_accum(self, seq, base_circuit, circuits)
        return circuits

    def __make_circuits_accum(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit,
                              accumulator: List[Tuple[QuantumCircuit, str]]):
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
        luder = luder_measurement(b, len(circuit.qubits) - 1, len(circuit.clbits), seq.depth - 1)
        circuit += luder

        # traversing deeper into the seq structure
        if seq.partitioning_measured is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_measured, circuit_copy, accumulator)
        else:
            accumulator.append((copy.deepcopy(circuit), seq.label))

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
