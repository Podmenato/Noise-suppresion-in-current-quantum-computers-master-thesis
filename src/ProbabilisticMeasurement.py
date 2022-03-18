import copy
from typing import List
import numpy as np
from POVM import POVM, Effect
from utilities import get_rotation_gate
import qiskit
from qiskit import *


class ProbabilityProjector:
    def __init__(self, vector: np.array, probability):
        self.vector = vector
        self.probability = probability
        self.shots = 1000


class ProbabilisticProjectiveMeasurement:
    def __init__(self, projectors: List[ProbabilityProjector]):
        self.projectors = projectors
        self.base1 = projectors[0].vector
        self.base2 = projectors[1].vector
        self.unitary = get_rotation_gate(self.base1, self.base2)

    def measure(self, circuit: QuantumCircuit):
        qasm = qiskit.Aer.get_backend("qasm_simulator")

        circuit1 = copy.deepcopy(circuit)
        circuit1.append(self.unitary, [circuit.qubits[0]])
        circuit1.measure(0, 0)
        circuit2 = copy.deepcopy(circuit)
        circuit2.append(self.unitary, [circuit.qubits[0]])
        circuit2.measure(0, 0)

        job1 = qiskit.execute(circuit1, qasm, shots=self.projectors[0].shots)
        job2 = qiskit.execute(circuit2, qasm, shots=self.projectors[1].shots)

        results1 = 0
        results2 = 0

        if self.projectors[0].probability != 0 and job1.result().get_counts().get('0') is not None:
            results1 = job1.result().get_counts().get('0')

        if self.projectors[1].probability != 0 and job2.result().get_counts().get('1') is not None:
            results2 = job2.result().get_counts().get('1')

        return results1, results2


class ProbabilisticMeasurement:
    def __init__(self, elements: List[np.array], labels=None):
        self.povms = POVM(elements, labels)
        self.projective_measurements = []

        for e in self.povms.elements:
            projectors = self.extract_projective_measurement(e)
            projective_measurement = ProbabilisticProjectiveMeasurement(projectors)
            self.projective_measurements.append(projective_measurement)

        executed_shots = 2000
        shots = 0
        probabilities = []

        for x in self.projective_measurements:
            x.projectors[0].shots = np.floor(x.projectors[0].shots*x.projectors[0].probability)
            x.projectors[1].shots = np.floor(x.projectors[1].shots*x.projectors[1].probability)
            shots += x.projectors[0].shots
            shots += x.projectors[1].shots
            probabilities.append(x.projectors[0].probability*(1/2))
            probabilities.append(x.projectors[1].probability*(1/2))

        additional_shots = np.random.multinomial(executed_shots - shots, probabilities)

        for i in range(len(additional_shots)):
            projective_measurement_idx = i // 2
            projector_idx = i % 2
            self.projective_measurements[projective_measurement_idx].projectors[projector_idx].shots += additional_shots[i]

        shots = 0
        for x in self.projective_measurements:
            shots += x.projectors[0].shots
            shots += x.projectors[1].shots

    def extract_projective_measurement(self, effect: Effect):
        u, d, v = np.linalg.svd(effect.matrix, full_matrices=True)

        projectors = []
        for i in range(2):
            uv = u[:, i]
            vv = v[i, :]
            prob_projector = ProbabilityProjector(uv, np.round(d[i], 14))
            projectors.append(prob_projector)

        return projectors

    # def distribute_probabilities(self):


    def measure(self, circuit: QuantumCircuit):
        results = []
        for meas in self.projective_measurements:
            r1, r2 = meas.measure(circuit)
            results.append(r1)
            results.append(r2)

        return results
00