from collections import Counter

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

from src.POVM import POVM
from src.ProbabilisticMeasurement import ProbabilisticMeasurement
from src.SequentialPOVMMeasurement import SequentialPOVMMeasurement
from src.utilities import simple_povm_xyz, povm_bell, scale_noise, save_variation_distances, load_variation_distances, \
    vd_int
import numpy as np

if __name__ == '__main__':
    omega = 0.5

    lamb = (np.cos(omega) ** 2) / 2

    phi1 = np.cos(omega) * np.array([1, 0]) + np.sin(omega) * np.array([0, 1])

    phi2 = np.cos(omega) * np.array([1, 0]) - np.sin(omega) * np.array([0, 1])

    phi1T = np.sin(omega) * np.array([1, 0]) - np.cos(omega) * np.array([0, 1])

    phi2T = np.sin(omega) * np.array([1, 0]) + np.cos(omega) * np.array([0, 1])

    A1 = (1 / 2) * np.array([[np.tan(omega) ** 2, np.tan(omega)], [np.tan(omega), 1]])

    A2 = (1 / 2) * np.array([[np.tan(omega) ** 2, -np.tan(omega)], [-np.tan(omega), 1]])

    A_u = np.array([[1 - np.tan(omega) ** 2, 0], [0, 0]])

    POVM([A1, A2, A_u]).validation()

    seq = SequentialPOVMMeasurement([A1, A2, A_u], ["A1", "A2", "A?"])
    # Prepare measured state
    prepared_state = QuantumCircuit(1, 1)
    initial_state = phi2 / np.linalg.norm(phi2)
    prepared_state.initialize(initial_state, 0)
    circs = seq.make_circuits([["A1", "A2"], ["A?"]], prepared_state)
    print(circs)
    print(seq.measure([["A1", "A2"], ["A?"]], prepared_state))

    seq = SequentialPOVMMeasurement(simple_povm_xyz, ["x+", "x-", "y+", "y-", "z+", "z-"])
    state = QuantumCircuit(1, 1)
    circs = seq.make_circuits([[["x+", "x-"], ["y+"]], [["z+", "z-"], ["y-"]]], state)
    print(circs)
    print(seq.measure([[["x+", "x-"], ["y+"]], [["z+", "z-"], ["y-"]]], state))
    print(seq.measure([["z+", "z-"], [["y+", "y-"], ["x+", "x-"]]], state))
