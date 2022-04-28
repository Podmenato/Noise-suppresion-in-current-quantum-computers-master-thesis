import copy
from typing import List
import numpy as np
from POVM import POVM, Effect
from utilities import get_rotation_gate, plot_results_histogram, map_number_to_qubit_result
import qiskit
from qiskit import *
from qiskit.providers.backend import Backend


class ProbabilityProjector:
    """
    Class representing a projector used in probabilistic measurements

    Attributes:
        - vector
        - probability
        - shots
    """

    def __init__(self, vector: np.array, probability):
        """
        Constructor for the class
        :param vector: basis vector
        :param probability: probability with which the projective measurement should be executed
        """
        self.vector = vector
        self.probability = probability
        self.shots = 1000


class ProbabilisticProjectiveMeasurement:
    """
    Class representing a projective measurement used in probabilistic measurements

    Attributes:
        - projectors - set of projectors
        - bases
        - unitary - the gate which rotates the state from computational basis, into this basis

    Methods:
        - measure
    """

    def __init__(self, projectors: List[ProbabilityProjector]):
        """
        Constructor for the class
        :param projectors: set of projectors of this measurement
        """
        self.projectors = projectors
        self.bases = []
        for projector in projectors:
            self.bases.append(projector.vector)
        self.unitary = get_rotation_gate(self.bases)

    def measure(self, circuit: QuantumCircuit, backend: Backend):
        """
        Performs a measurement in the probabilistic measurement process
        :param circuit: the measured state
        :param backend: optional backend to be executed on
        :return: count of measurements
        """
        circuits = []
        for i in range(2 ** len(circuit.qubits)):
            circ = copy.deepcopy(circuit)
            circ.append(self.unitary, circuit.qubits)
            circ.measure_all(add_bits=False)
            circuits.append(circ)

        jobs = []
        for i in range(len(circuits)):
            job = qiskit.execute(circuits[i], backend, shots=int(self.projectors[i].shots))
            jobs.append(job)

        results = [0 for _ in range(len(self.projectors))]

        for i in range(len(self.projectors)):
            current_result = map_number_to_qubit_result(i, len(circuit.qubits))
            if self.projectors[i].probability != 0 and jobs[i].result().get_counts().get(current_result) is not None:
                results[i] = jobs[i].result().get_counts().get(current_result)

        return np.sum(results)


class ProbabilisticMeasurement:
    """
    Class representing the probabilistic POVM measurement

    Attributes:
    """
    def __init__(self, elements: List[np.array], labels=None):
        """
        Constructor for the class
        :param elements: Effects of the POVM
        :param labels: Labels for the effects
        """
        self.povm = POVM(elements, labels)
        self.projective_measurements = []

        for e in self.povm.elements:
            projectors = self.extract_projective_measurement(e)
            projective_measurement = ProbabilisticProjectiveMeasurement(projectors)
            self.projective_measurements.append(projective_measurement)

    @staticmethod
    def extract_projective_measurement(effect: Effect):
        u, d, v = np.linalg.svd(effect.matrix, full_matrices=True)

        projectors = []
        for i in range(effect.matrix.shape[0]):
            uv = u[:, i]
            vv = v[i, :]
            prob_projector = ProbabilityProjector(uv, np.round(d[i], 14))
            projectors.append(prob_projector)

        return projectors

    def measure(self, circuit: QuantumCircuit, shots=1000, backend=qiskit.Aer.get_backend("qasm_simulator")):
        """
        Performs an probabilistic measurement POVM simulation
        :param circuit: measured state
        :param shots: numbers of executed shots
        :param backend: optional backend to be executed on
        :return: Results count
        """

        executed_shots = shots
        shots_per_measurement = shots/len(self.projective_measurements[0].projectors)

        shots = 0
        probabilities = []

        for x in self.projective_measurements:
            for projector in x.projectors:
                projector.shots = np.floor(shots_per_measurement * projector.probability)
                shots += projector.shots
                probabilities.append(projector.probability * (1 / len(x.projectors)))

        additional_shots = np.random.multinomial(executed_shots - shots, probabilities)

        for i in range(len(additional_shots)):
            projective_measurement_idx = i // len(self.projective_measurements[0].projectors)
            projector_idx = i % len(self.projective_measurements[0].projectors)
            self.projective_measurements[projective_measurement_idx].projectors[projector_idx].shots \
                += additional_shots[i]

        shots = 0
        for x in self.projective_measurements:
            for projector in x.projectors:
                shots += projector.shots

        results = []
        for meas in self.projective_measurements:
            r = meas.measure(circuit, backend)
            results.append(r)

        return results

    def plot_histogram(self, results: List[int], title=None, shots=1000) -> None:
        """
        Plots a histogram of the Probabilistic POVM results, with the corresponding labels.
        Transforms the measurement counts to percentages.

        :param results: of the measurement
        :param shots: number of performed shots in the measurements
        :param title: optional title of the histogram
        :return: None
        """

        percentages = []
        for result in results:
            percentages.append(result / shots)

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        plot_results_histogram(percentages, labels, title)
