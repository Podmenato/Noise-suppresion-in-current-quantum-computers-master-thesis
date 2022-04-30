import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

from src.ProbabilisticMeasurement import ProbabilisticMeasurement
from src.SequentialPOVMMeasurement import SequentialPOVMMeasurement
from src.utilities import simple_povm_xyz, povm_bell, scale_noise, save_variation_distances, load_variation_distances, vd_int
from collections import Counter
import numpy as np

if __name__ == '__main__':
    qasm = qiskit.Aer.get_backend("qasm_simulator")
    seq = SequentialPOVMMeasurement(simple_povm_xyz, ["x+", "x-", "y+", "y-", "z+", "z-"])
    prob = ProbabilisticMeasurement(simple_povm_xyz, ["x+", "x-", "y+", "y-", "z+", "z-"])
    ideal_results = [166, 166, 166, 166, 333, 0]

    qiskit.IBMQ.load_account()
    # Device to be simulated
    dev = qiskit.IBMQ.get_provider().get_backend("ibmq_manila")

    # Extracted simulator for the device
    sim = AerSimulator.from_backend(dev)

    # noise model of the simulator
    state = QuantumCircuit(1, 1)

    scales = np.linspace(0, 1, 15)

    for i in range(20):
        vd1 = []
        vd5 = []
        vd10 = []
        vd15 = []
        vd20 = []
        for scale in scales:
            noise_model = scale_noise(NoiseModel().from_backend(dev), scale)
            sim_noise = AerSimulator(noise_model=noise_model)
            sequence, dictionary = seq.measure_result_sequence(
                [["z+", "z-"], [["y+", "y-"], ["x+", "x-"]]], state, backend=sim_noise, shots=20000)
            res1k = seq.parse_sequence_results(sequence, dictionary, shots=1000)
            res5k = seq.parse_sequence_results(sequence, dictionary, shots=5000)
            res10k = seq.parse_sequence_results(sequence, dictionary, shots=10000)
            res15k = seq.parse_sequence_results(sequence, dictionary, shots=15000)
            res20k = seq.parse_sequence_results(sequence, dictionary, shots=20000)

            # jobs = prob.measure(state, backend=sim_noise, shots=20000, memory=True)
            # res1k = prob.parse_sequences(jobs, executed_shots=20000, parsed_shots=1000, qubits=1)
            # res5k = prob.parse_sequences(jobs, executed_shots=20000, parsed_shots=5000, qubits=1)
            # res10k = prob.parse_sequences(jobs, executed_shots=20000, parsed_shots=10000, qubits=1)
            # res15k = prob.parse_sequences(jobs, executed_shots=20000, parsed_shots=15000, qubits=1)
            # res20k = prob.parse_sequences(jobs, executed_shots=20000, parsed_shots=20000, qubits=1)

            vd1.append(vd_int(res1k, ideal_results))
            vd5.append(vd_int(res5k, ideal_results))
            vd10.append(vd_int(res10k, ideal_results))
            vd15.append(vd_int(res15k, ideal_results))
            vd20.append(vd_int(res20k, ideal_results))
        save_variation_distances(vd1, "output/z_meas_seq_1000.txt")
        save_variation_distances(vd5, "output/z_meas_seq_5000.txt")
        save_variation_distances(vd10, "output/z_meas_seq_10000.txt")
        save_variation_distances(vd15, "output/z_meas_seq_15000.txt")
        save_variation_distances(vd20, "output/z_meas_seq_20000.txt")
        # save_variation_distances(vd1, "output/z_meas_prob_1000.txt")
        # save_variation_distances(vd5, "output/z_meas_prob_5000.txt")
        # save_variation_distances(vd10, "output/z_meas_prob_10000.txt")
        # save_variation_distances(vd15, "output/z_meas_prob_15000.txt")
        # save_variation_distances(vd20, "output/z_meas_prob_20000.txt")

print("Done computing!")
