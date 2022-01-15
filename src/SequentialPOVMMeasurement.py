import copy
from typing import List
import numpy as np
from qiskit import *
from utilities import measurement_change, luder_measurement
from POVM import POVM, Effect
import matplotlib.pyplot as plt


class SequentialPOVMMeasurementTree:
    def __init__(self, elements: List[Effect], partitioning: list, depth=0, measuring_qubits="") -> None:
        """
        Internal structure representing the sequential POVM measurement tree, based on the entered partitioning
        :param elements: list of all element matrices
        :param partitioning: list of indices, representing the partitioning, i.e. [[1-2],[3-4]]
        :param depth: depth of the node inside the tree
        """
        self.depth = depth + 1
        labels1, labels2 = SequentialPOVMMeasurementTree.__get_results_idx_list(partitioning)

        effects1 = []
        effects2 = []

        for element in elements:
            if element.label in labels1:
                effects1.append(element)

            if element.label in labels2:
                effects2.append(element)

        self.result_measured = effects1
        self.result_other = effects2

        self.measuring_qubits = '1' + measuring_qubits
        self.measuring_qubits_other = '0' + measuring_qubits

        self.partitioning_measured = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects1, labels1,
                                                                                           partitioning[0],
                                                                                           depth=self.depth,
                                                                                           measuring_qubits=self.measuring_qubits)

        self.partitioning_other = SequentialPOVMMeasurementTree.__make_sub_measurements(elements, effects2, labels2,
                                                                                        partitioning[1],
                                                                                        depth=self.depth,
                                                                                        measuring_qubits=self.measuring_qubits_other)

    @staticmethod
    def __get_results_idx_list(partitioning: list):
        if len(partitioning) != 2:
            raise Exception("Invalid partitioning format")

        if type(partitioning[0]) is not type(partitioning[1]):
            raise Exception("Invalid partitioning format")

        if not isinstance(partitioning[0], list):
            return [partitioning[0]], [partitioning[1]]

        p1 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[0])
        p2 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[1])

        return p1, p2

    @staticmethod
    def __flatten_partitioning(partitioning: list):
        if len(partitioning) > 2 | len(partitioning) == 0:
            raise Exception("Invalid partitioning format")

        if len(partitioning) == 1:
            if not isinstance(partitioning[0], list):
                return partitioning
            else:
                raise Exception("Invalid partitioning format")

        if type(partitioning[0]) is not type(partitioning[1]):
            raise Exception("Invalid partitioning format")

        if not isinstance(partitioning[0], list):
            return partitioning
        elif isinstance(partitioning[0], list):
            p1 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[0])
            p2 = SequentialPOVMMeasurementTree.__flatten_partitioning(partitioning[1])
            return [*p1, *p2]

    @staticmethod
    def __make_sub_measurements(all_elements: List[Effect], measured_elements: List[Effect], labels: list,
                                partitioning: list, depth: int, measuring_qubits: str):
        measured = []
        for x in measured_elements:
            measured.append(x.matrix)
        if len(measured_elements) != 1:
            new_elements = []
            for element in all_elements:
                if element.label in labels:
                    b = np.sum(measured, 0)
                    change = measurement_change(b, element.matrix)
                    new_elements.append(Effect(change, element.label))
            return SequentialPOVMMeasurementTree(new_elements, partitioning, depth, measuring_qubits=measuring_qubits)
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


class SequentialPOVMMeasurementCircuit:
    def __init__(self, q_circuit: QuantumCircuit, one: Effect, zero: Effect, one_result: str, zero_result: str):
        """
        Class representing the output of the make_circuits method
        :param q_circuit: The quantum circuit measuring the corresponding effects
        :param one: Effect when qubit has value 1
        :param zero: Effect when qubit has value 0
        """
        self.q_circuit = q_circuit
        self.one: Effect = one
        self.zero: Effect = zero

        self.one_result = one_result
        self.zero_result = zero_result

    def plot_histogram(self, data, title=None) -> None:
        """
        Plots the histogram using results of an experiment and assigns the counts
        to the corresponding buckets based on the class attributes
        :param data: Dictionary, results of a qiskit job execution
        :param title: Label for the plot
        :return: None, only plots histogram
        """
        keys = list(data.keys())

        names = [str(self.one.label), str(self.zero.label), 'rest']
        key_qubits = [self.one_result, self.zero_result]

        rest = 0
        for key in keys:
            if key not in key_qubits:
                rest += data[key]

        measured = 0
        if key_qubits[0] in keys:
            measured = data[key_qubits[0]]

        measured_opposite = 0
        if key_qubits[1] in keys:
            measured_opposite = data[key_qubits[1]]

        values = [measured / 1000, measured_opposite / 1000, rest / 1000]

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


class SequentialPOVMMeasurement:
    def __init__(self, elements: List[np.array], labels: list) -> None:
        """
        Class used to create circuits for POVM representation using the sequential measurement method
        :param elements: list of all element matrices
        :param labels: TODO
        """
        self.povm = POVM(elements, labels)

    def make_circuits(self, partitioning: list, state: QuantumCircuit) -> List[SequentialPOVMMeasurementCircuit]:
        """
        Creates circuits, each measures its specific measurement
        :param state: circuit with prepared state
        :param partitioning: list of indices, each corresponding to the index in the element in elements array.
            Should be in the following format [[1, 2],[3, 4]]. Note, [1,[2, 3]] is not valid, [[1], [2, 3]] is.
        :return: List of QuantumCircuits
        """
        seq = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)
        circuits: List[SequentialPOVMMeasurementCircuit] = []
        qubits_count = len(state.qubits)
        classical_count = seq.largest_depth()
        base_circuit = QuantumCircuit(qubits_count + 1, classical_count)
        base_circuit = base_circuit.compose(state)
        SequentialPOVMMeasurement.__make_circuits_accum(self, seq, base_circuit, circuits)

        # Add zero in case some circuit is shorter than others,
        # in order to match the results of the experiment
        for circ in circuits:
            diff = (classical_count - len(circ.one_result))
            if diff > 0:
                circ.one_result = ('0'*diff) + circ.one_result
                circ.zero_result = ('0'*diff) + circ.zero_result

        return circuits

    def __make_circuits_accum(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit,
                              accumulator: List[SequentialPOVMMeasurementCircuit]):
        """
        Helper method used to traverse the SequentialPOVMMeasurementTree and create circuits from it
        :param seq: Internal tree structure
        :param circuit: quantum circuit on to which new addition should be appended
        :param accumulator: list of already finished circuits
        :return: None
        """

        # current measurement
        rslt_msrd = []
        for x in seq.result_measured:
            rslt_msrd.append(x.matrix)
        b = np.sum(rslt_msrd, 0)

        # create luder measurement circuit based on b
        luder = luder_measurement(b, len(circuit.qubits) - 1, len(circuit.clbits), seq.depth - 1)
        circuit = circuit.compose(luder)

        # traversing deeper into the seq structure
        if seq.partitioning_measured is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_measured, circuit_copy, accumulator)
        else:
            accumulator.append(
                SequentialPOVMMeasurementCircuit(copy.deepcopy(circuit), seq.result_measured[0], seq.result_other[0],
                                                 seq.measuring_qubits, seq.measuring_qubits_other))

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
        seq = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)

    def __make_single_circuit_helper(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit):
        """
        WIP
        :param seq:
        :param circuit:
        :return:
        """
        # TODO implement method
        b = np.sum(seq.result_measured, 0)
