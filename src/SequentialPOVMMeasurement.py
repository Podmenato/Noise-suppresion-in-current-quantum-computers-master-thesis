import copy
from typing import List, Dict, Tuple
import numpy as np
import qiskit
from qiskit import *
from utilities import measurement_change, luder_measurement, plot_results_histogram, luder_measurement_single_circuit
from POVM import POVM, Effect
from collections import Counter
import matplotlib.pyplot as plt
from random import shuffle


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


class SequentialPOVMMeasurementSequence:
    def __init__(self, measurement_results: List[str], result_one: str, result_zero: str) -> None:
        self.measurement_results = measurement_results
        self.result_one = result_one
        self.result_zero = result_zero


class SequentialPOVMMeasurement:
    def __init__(self, elements: List[np.array], labels=None) -> None:
        """
        Class used to create circuits for POVM representation using the sequential measurement method
        :param elements: list of all element matrices
        :param labels: TODO
        """
        self.povm = POVM(elements, labels)
        self.result_labels = []

    def make_circuits(self, partitioning: list, state: QuantumCircuit, real_device=False) \
            -> List[SequentialPOVMMeasurementCircuit]:
        """
        Creates circuits, each measures its specific measurement
        :param real_device: whether the circuit should be prepared to work on a real device
        :param state: circuit with prepared state
        :param partitioning: list of indices, each corresponding to the index in the element in elements array.
            Should be in the following format [[1, 2],[3, 4]]. Note, [1,[2, 3]] is not valid, [[1], [2, 3]] is.
        :return: List of QuantumCircuits
        """
        seq = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)
        circuits: List[SequentialPOVMMeasurementCircuit] = []
        qubits_count = len(state.qubits)
        classical_count = seq.largest_depth()
        if real_device:
            base_circuit = QuantumCircuit(qubits_count + classical_count, classical_count)
        else:
            base_circuit = QuantumCircuit(qubits_count + 1, classical_count)
        base_circuit = base_circuit.compose(state)
        SequentialPOVMMeasurement.__make_circuits_accum(self, seq, base_circuit, len(state.qubits), circuits,
                                                        real_device=real_device)

        # Add zero in case some circuit is shorter than others,
        # in order to match the results of the experiment
        for circ in circuits:
            diff = (classical_count - len(circ.one_result))
            if diff > 0:
                circ.one_result = ('0' * diff) + circ.one_result
                circ.zero_result = ('0' * diff) + circ.zero_result

        return circuits

    def __make_circuits_accum(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit, state_qubits: int,
                              accumulator: List[SequentialPOVMMeasurementCircuit], real_device=False):
        """
        Helper method used to traverse the SequentialPOVMMeasurementTree and create circuits from it
        :param seq: Internal tree structure
        :param circuit: quantum circuit on to which new addition should be appended
        :param state_qubits: number of qubits the state is on
        :param accumulator: list of already finished circuits
        :param real_device: whether the circuit should be prepared to work on a real device
        :return: None
        """

        # current measurement
        rslt_msrd = []
        for x in seq.result_measured:
            rslt_msrd.append(x.matrix)
        b = np.sum(rslt_msrd, 0)

        measuring_qubit = None
        if real_device:
            measuring_qubit = state_qubits + seq.depth - 1

        # create luder measurement circuit based on b
        luder = luder_measurement(b,
                                  len(circuit.qubits),
                                  state_qubits,
                                  len(circuit.clbits),
                                  seq.depth - 1,
                                  real_device=real_device, measuring_qubit=measuring_qubit)
        circuit = circuit.compose(luder)

        # traversing deeper into the seq structure
        if seq.partitioning_measured is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_measured, circuit_copy, state_qubits,
                                                            accumulator, real_device=real_device)
        else:
            accumulator.append(
                SequentialPOVMMeasurementCircuit(copy.deepcopy(circuit), seq.result_measured[0], seq.result_other[0],
                                                 seq.measuring_qubits, seq.measuring_qubits_other))

        if seq.partitioning_other is not None:
            circuit_copy = copy.deepcopy(circuit)
            SequentialPOVMMeasurement.__make_circuits_accum(self, seq.partitioning_other, circuit_copy, state_qubits,
                                                            accumulator, real_device=real_device)

    def measure(self, partitioning: list, state: QuantumCircuit, shots=1000, backend=None, real_device=False) -> List[
        int]:
        """
        Measures all the circuits from the sequential measurements and puts the results
        together.

        :param partitioning: list of indices, each corresponding to the index in the element in elements array.
            Should be in the following format [[1, 2],[3, 4]]. Note, [1,[2, 3]] is not valid, [[1], [2, 3]] is.
        :param state: circuit with prepared state
        :param shots: optional number of shots
        :param backend: optional backend the circuits should be executed on
        :param real_device: whether the circuit should be prepared to work on a real device
        :return: list of result counts
        """

        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        circuits = self.make_circuits(partitioning, state, real_device=real_device)

        divided_shots = np.floor(shots / len(circuits))
        probabilities = [1 / len(circuits) for _ in range(len(circuits))]
        additional_shots = np.random.multinomial(shots - divided_shots * len(circuits), probabilities)

        results = [0] * len(self.povm.elements)

        for i in range(len(circuits)):
            job = qiskit.execute(circuits[i].q_circuit, backend, shots=int(divided_shots + additional_shots[i]))
            data = job.result().get_counts()

            keys = list(data.keys())

            result1 = 0
            if circuits[i].one_result in keys:
                result1 = data[circuits[i].one_result]
            results[labels.index(circuits[i].one.label)] = result1

            result2 = 0
            if circuits[i].zero_result in keys:
                result2 = data[circuits[i].zero_result]
            results[labels.index(circuits[i].zero.label)] = result2

        return results

    def measure_result_sequence(self, partitioning: list, state: QuantumCircuit, shots=1000,
                                backend=None, real_device=False) -> Tuple[
        List[SequentialPOVMMeasurementSequence], Dict[str, str]]:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        circuits = self.make_circuits(partitioning, state, real_device=real_device)

        divided_shots = np.floor(shots / len(circuits))
        probabilities = [1 / len(circuits) for _ in range(len(circuits))]
        additional_shots = np.random.multinomial(shots - divided_shots * len(circuits), probabilities)

        tree = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)
        dictionary = dict()
        self.__make_label_result_dictionary(tree, tree.largest_depth(), 1, dictionary)

        results = []
        for i in range(len(circuits)):
            job = qiskit.execute(circuits[i].q_circuit, backend, shots=divided_shots + additional_shots[i], memory=True)
            data = job.result().get_memory()
            results.append(SequentialPOVMMeasurementSequence(data, circuits[i].one_result, circuits[i].zero_result))

        return results, dictionary

    def parse_sequence_results(self, data: List[SequentialPOVMMeasurementSequence],
                               label_result_dictionary: Dict[str, str], shots=None):
        if shots is None:
            shots = 0
            for x in data:
                shots += len(x.measurement_results)

        divided_shots = np.floor(shots / len(data))
        probabilities = [1 / len(data) for _ in range(len(data))]
        additional_shots = np.random.multinomial(shots - divided_shots * len(data), probabilities)

        sampled_data = []
        for i in range(len(data)):
            sample = data[i].measurement_results[0:int((divided_shots+additional_shots[i]))]
            filtered = filter(lambda s: s == data[i].result_one or s == data[i].result_zero, sample)
            sampled_data.extend(filtered)

        counter = Counter(sampled_data)

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        results = [0] * len(labels)

        keys = list(counter.keys())

        for key in keys:
            label = label_result_dictionary[key]
            idx = labels.index(label)
            results[idx] = counter[key]

        return results


    def measure_single_circuit(self, partitioning: list, state: QuantumCircuit, shots=1000, backend=None) -> List[int]:
        """
        Builds a single circuit measuring sequential measurement based on previous results using c_if condition.
        This means this method does not work on a real hardware, only on simulator.
        :param partitioning: list of indices, each corresponding to the index in the element in elements array.
            Should be in the following format [[1, 2],[3, 4]]. Note, [1,[2, 3]] is not valid, [[1], [2, 3]] is.
        :param state: circuit with prepared state
        :param shots: optional number of shots
        :param backend: optional backend the circuits should be executed on
        :return: list of result counts
        """

        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        results = [0] * len(labels)

        self.result_labels = []

        circuit = self.make_single_circuit(partitioning, state)

        job = qiskit.execute(circuit, backend, shots=shots)
        data = job.result().get_counts()

        keys = list(data.keys())

        for result_label in self.result_labels:
            if result_label[0] in keys:
                idx = labels.index(result_label[1])
                results[idx] = data[result_label[0]]

        return results

    def measure_result_sequence_single_circuit(self, partitioning: list, state: QuantumCircuit, shots=1000,
                                               backend=None) -> Tuple[List[str], Dict[str, str]]:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        circuit = self.make_single_circuit(partitioning, state)

        job = qiskit.execute(circuit, backend, shots=shots, memory=True)
        data = job.result().get_memory()

        tree = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)
        dictionary = dict()
        self.__make_label_result_dictionary(tree, tree.largest_depth(), 1, dictionary)

        return data, dictionary

    def parse_sequence_results_single_circuit(self, data: List[str], label_result_dictionary: Dict[str, str],
                                              shots=None):
        if shots is None:
            shots = len(data)

        results = data[0:shots]

        counter = Counter(results)

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        results = [0] * len(labels)

        keys = list(counter.keys())
        label_keys = list(label_result_dictionary.keys())

        for key in keys:
            if key not in label_keys:
                continue
            label = label_result_dictionary[key]
            idx = labels.index(label)
            results[idx] = counter[key]

        return results

    def __make_label_result_dictionary(self, tree: SequentialPOVMMeasurementTree, max_depth, depth, dic):
        diff = max_depth - depth

        if len(tree.result_measured) == 1:
            dic[('0' * diff) + tree.measuring_qubits] = tree.result_measured[0].label

        if len(tree.result_other) == 1:
            dic[('0' * diff) + tree.measuring_qubits_other] = tree.result_other[0].label

        if tree.partitioning_measured is not None:
            self.__make_label_result_dictionary(tree.partitioning_measured, max_depth, depth + 1, dic=dic)

        if tree.partitioning_other is not None:
            self.__make_label_result_dictionary(tree.partitioning_other, max_depth, depth + 1, dic=dic)

    def plot_histogram(self, results: List[int], title=None) -> None:
        """
        Plots a histogram of the Sequential POVM results, with the corresponding labels.
        Transforms the measurement counts to percentages.

        :param results: of the measurement
        :param title: optional title of the histogram
        :return: None
        """
        total = np.sum(results)
        percentages = []
        for result in results:
            percentages.append(np.round((result / total), 3))

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        plot_results_histogram(percentages, labels, title)

    def make_single_circuit(self, partitioning: list, state: QuantumCircuit) -> QuantumCircuit:
        seq = SequentialPOVMMeasurementTree(self.povm.elements, partitioning)
        qubits_count = len(state.qubits)
        classical_count = seq.largest_depth()
        q_reg = QuantumRegister(qubits_count + 1)
        c_reg = ClassicalRegister(classical_count)
        base_circuit = QuantumCircuit(q_reg, c_reg)
        base_circuit = base_circuit.compose(state)
        return SequentialPOVMMeasurement.__make_single_circuit_helper(self, seq, base_circuit, c_reg, classical_count)

    def __make_single_circuit_helper(self, seq: SequentialPOVMMeasurementTree, circuit: QuantumCircuit,
                                     c_reg: ClassicalRegister,
                                     largest_depth: int) -> QuantumCircuit:
        measurements = []
        self.__traverse_seq_tree_collect_meas(seq, 1, largest_depth, '', measurements)

        measurements.sort(key=lambda m: len(m[1]))

        current_measurements = []
        for meas in measurements:
            if len(current_measurements) == 0 or len(meas[1]) == len(current_measurements[0][1]):
                current_measurements.append(meas)
                continue
            luder_measurement_single_circuit(current_measurements, len(circuit.qubits) - 1, circuit, c_reg,
                                             len(current_measurements[0][1]) - 1)
            current_measurements = [meas]

        luder_measurement_single_circuit(current_measurements, len(circuit.qubits) - 1, circuit, c_reg,
                                         len(current_measurements[0][1]) - 1)
        return circuit

    def __traverse_seq_tree_collect_meas(self,
                                         seq: SequentialPOVMMeasurementTree,
                                         depth: int,
                                         max_depth: int,
                                         condition: str,
                                         accumulator: List[Tuple[np.array, str, str]]) -> None:
        b = []
        for effect in seq.result_measured:
            b.append(effect.matrix)

        accumulator.append((np.sum(b, 0), seq.measuring_qubits, condition))

        diff = max_depth - depth

        if len(seq.result_measured) == 1:
            self.result_labels.append((('0' * diff) + seq.measuring_qubits, seq.result_measured[0].label))

        if len(seq.result_other) == 1:
            self.result_labels.append((('0' * diff) + seq.measuring_qubits_other, seq.result_other[0].label))

        if seq.partitioning_measured is not None:
            self.__traverse_seq_tree_collect_meas(seq.partitioning_measured, depth + 1, max_depth, '1' + condition,
                                                  accumulator)

        if seq.partitioning_other is not None:
            self.__traverse_seq_tree_collect_meas(seq.partitioning_other, depth + 1, max_depth, '0' + condition,
                                                  accumulator)

        return None
