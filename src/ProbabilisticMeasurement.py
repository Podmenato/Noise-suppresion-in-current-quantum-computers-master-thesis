import copy
from typing import List
import numpy as np
from POVM import POVM, Effect
from utilities import get_rotation_gate, plot_results_histogram, map_number_to_qubit_result
import qiskit
from qiskit import *
from qiskit.providers.backend import Backend


class ProbabilityProjector:
    def __init__(self, vector: np.array, probability):
        self.vector = vector
        self.probability = probability
        self.shots = 1000


class ProbabilisticProjectiveMeasurement:
    def __init__(self, projectors: List[ProbabilityProjector]):
        self.projectors = projectors
        self.bases = []
        for projector in projectors:
            self.bases.append(projector.vector)
        self.unitary = get_rotation_gate(self.bases)

    def measure(self, circuit: QuantumCircuit, backend: Backend):
        circuits = []
        for i in range(2**len(circuit.qubits)):
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
    def __init__(self, elements: List[np.array], labels=None, backend=None):
        self.povm = POVM(elements, labels)
        self.projective_measurements = []
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        if backend is not None:
            self.backend = backend

        for e in self.povm.elements:
            projectors = self.extract_projective_measurement(e)
            projective_measurement = ProbabilisticProjectiveMeasurement(projectors)
            self.projective_measurements.append(projective_measurement)

        executed_shots = 1000*len(self.projective_measurements[0].projectors)
        shots = 0
        probabilities = []

        for x in self.projective_measurements:
            for projector in x.projectors:
                projector.shots = np.floor(projector.shots*projector.probability)
                shots += projector.shots
                probabilities.append(projector.probability*(1/len(x.projectors)))

        additional_shots = np.random.multinomial(executed_shots - shots, probabilities)

        for i in range(len(additional_shots)):
            projective_measurement_idx = i // len(self.projective_measurements[0].projectors)
            projector_idx = i % len(self.projective_measurements[0].projectors)
            self.projective_measurements[projective_measurement_idx].projectors[projector_idx].shots += additional_shots[i]

        shots = 0
        for x in self.projective_measurements:
            for projector in x.projectors:
                shots += projector.shots

    def extract_projective_measurement(self, effect: Effect):
        u, d, v = np.linalg.svd(effect.matrix, full_matrices=True)

        projectors = []
        for i in range(effect.matrix.shape[0]):
            uv = u[:, i]
            vv = v[i, :]
            prob_projector = ProbabilityProjector(uv, np.round(d[i], 14))
            projectors.append(prob_projector)

        return projectors

    def measure(self, circuit: QuantumCircuit):
        results = []
        for meas in self.projective_measurements:
            r = meas.measure(circuit, self.backend)
            results.append(r)

        return results

    def plot_histogram(self, results: List[int], title=None):
        percentages = []
        for result in results:
            percentages.append(result/1000)

        labels = []
        for element in self.povm.elements:
            labels.append(element.label)

        plot_results_histogram(percentages, labels, title)
