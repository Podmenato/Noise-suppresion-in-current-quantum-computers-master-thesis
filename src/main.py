import qiskit
from qiskit import QuantumCircuit

from src.ProbabilisticMeasurement import ProbabilisticMeasurement
from src.SequentialPOVMMeasurement import SequentialPOVMMeasurement
from src.utilities import simple_povm_xyz
from collections import Counter

if __name__ == '__main__':
    qasm = qiskit.Aer.get_backend("qasm_simulator")
    prob = ProbabilisticMeasurement(simple_povm_xyz, ["x+", "x-", "y+", "y-", "z+", "z-"])
    seq = SequentialPOVMMeasurement(simple_povm_xyz, ["x+", "x-", "y+", "y-", "z+", "z-"])
    state = QuantumCircuit(1, 1)
    results = prob.measure(state, shots=10000)
    print(results)

    sequence, dictionary = seq.measure_result_sequence([["z+", "z-"], [["y+", "y-"], ["x+", "x-"]]], state, shots=10000)
    results = seq.parse_sequence_results(sequence, dictionary, shots=1000)
    print(results)

    sequence, dictionary = seq.measure_result_sequence_single_circuit([["z+", "z-"], [["y+", "y-"], ["x+", "x-"]]], state, shots=10000)
    results = seq.parse_sequence_results_single_circuit(sequence, dictionary, shots=5000)
    print(results)
