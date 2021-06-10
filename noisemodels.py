from qiskit import *
from qiskit.providers.aer.noise import NoiseModel, QuantumError

import numpy as np

X = np.asarray([[0, 1], [1, 0]])
Y = np.asarray([[0, -1j], [1j, 0]])
Z = np.asarray([[1, 0], [0, -1]])
I = np.asarray([[1, 0], [0, 1]])

pauli_dict = {"X": X, "Y": Y, "Z": Z, "I": I}


def create_depolarizing_error_model(p_cnot: float, p_u: float, p_meas: float, n_qubits: int = 5):
    noise_model = NoiseModel()
    two_qubit_error_map = np.zeros((n_qubits, n_qubits))

    # Two-qubit depolarizing error channel with error rate p_cnot on CNOT-gates
    kraus_operators_cnot = [np.sqrt(1 - p_cnot) * np.kron(I, I)]

    if p_cnot > 0:
        for a in ["I", "X", "Y", "Z"]:
            for b in ["I", "X", "Y", "Z"]:
                if not ((a == "I") and (b == "I")):
                    op = np.kron(pauli_dict[a], pauli_dict[b])
                    kraus_operators_cnot.append(np.sqrt(p_cnot / 15) * op)

        cnot_error = QuantumError(noise_ops=kraus_operators_cnot, number_of_qubits=2)

        two_qubit_error_map += p_cnot

        noise_model.add_all_qubit_quantum_error(cnot_error, ["cx"])

    if p_u > 0:
        # One-qubit depolarizing error channel with error rate p_u on all single-qubit gates
        kraus_operators_u = [np.sqrt(1 - p_u) * I, np.sqrt(p_u / 3) * X, np.sqrt(p_u / 3) * Y, np.sqrt(p_u / 3) * Z]
        u_error = QuantumError(noise_ops=kraus_operators_u, number_of_qubits=1)

        noise_model.add_all_qubit_quantum_error(u_error, ["u"])

    if p_meas > 0:
        # One-qubit depolarizing error channel with error rate p_meas on all measurement gates
        kraus_operators_meas = [np.sqrt(1 - p_meas) * I, np.sqrt(p_meas / 3) * X, np.sqrt(p_meas / 3) * Y,
                                np.sqrt(p_meas / 3) * Z]
        meas_error = QuantumError(noise_ops=kraus_operators_meas, number_of_qubits=1)

        noise_model.add_all_qubit_quantum_error(meas_error, ["measure"])

    return noise_model, two_qubit_error_map

def create_amplitude_damping_error_model(p_cnot: float, p_u: float, n_qubits: int = 5):
    # Kraus operators for 1-qubit amplitude damping channel
    def get_kraus_ops(p):
        K_0 = np.asarray([[1, 0],
                          [0, np.sqrt(1-p)]])
        K_1 = np.asarray([[0, np.sqrt(p)],
                          [0, 0]])
        return [K_0, K_1]

    kraus_ops_u = get_kraus_ops(p_u)
    kraus_ops_cnot_singleq = get_kraus_ops(p_cnot)

    singleq_error = QuantumError(noise_ops=kraus_ops_u, number_of_qubits=1)

    cnot_singleq_error = QuantumError(noise_ops=kraus_ops_cnot_singleq, number_of_qubits=1)

    # To create a 2-qubit error we take the tensor product of two independent 1-qubit amplitude damping errors
    cnot_error = cnot_singleq_error.tensor(cnot_singleq_error)

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(singleq_error, ["u"])
    noise_model.add_all_qubit_quantum_error(cnot_error, ["cx"])

    # The tensor product of 2 independent error channels with error rate p will have an error rate of 2*p - p^2
    #cnot_error_rate = 2*p_cnot - p_cnot**2
    cnot_error_rate = p_cnot

    two_qubit_error_map = np.zeros((n_qubits, n_qubits)) + cnot_error_rate

    return noise_model, two_qubit_error_map

def create_phase_damping_error_model(p_cnot: float, p_u: float, n_qubits: int = 5):
    # Kraus operators for 1-qubit phase damping channel:
    def get_kraus_ops(p):
        K_0 = np.asarray([[np.sqrt(1-p), 0],
                          [0, np.sqrt(1-p)]])
        K_1 = np.asarray([[np.sqrt(p), 0],
                          [0, 0]])
        K_2 = np.asarray([[0, 0],
                          [0, np.sqrt(p)]])
        return [K_0, K_1, K_2]

    kraus_ops_u = get_kraus_ops(p_u)
    kraus_ops_cnot_singleq = get_kraus_ops(p_cnot)

    singleq_error = QuantumError(noise_ops=kraus_ops_u, number_of_qubits=1)

    cnot_singleq_error = QuantumError(noise_ops=kraus_ops_cnot_singleq, number_of_qubits=1)

    # To create a 2-qubit error we take the tensor product of two independent 1-qubit phase damping errors
    cnot_error = cnot_singleq_error.tensor(cnot_singleq_error)

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(singleq_error, ["u"])
    noise_model.add_all_qubit_quantum_error(cnot_error, ["cx"])

    #cnot_error_rate = 2 * p_cnot - p_cnot ** 2
    cnot_error_rate = p_cnot

    two_qubit_error_map = np.zeros((n_qubits, n_qubits)) + cnot_error_rate

    return noise_model, two_qubit_error_map

def create_bit_phase_flip_error_model(p_cnot: float, p_u: float, n_qubits: int = 5):
    # Kraus operators for 1-qubit phase damping channel:
    def get_kraus_ops(p):
        return [(1-p)*I, ]

    kraus_ops_u = get_kraus_ops(p_u)
    kraus_ops_cnot_singleq = get_kraus_ops(p_cnot)

    singleq_error = QuantumError(noise_ops=kraus_ops_u, number_of_qubits=1)

    cnot_singleq_error = QuantumError(noise_ops=kraus_ops_cnot_singleq, number_of_qubits=1)

    # To create a 2-qubit error we take the tensor product of two independent 1-qubit phase damping errors
    cnot_error = cnot_singleq_error.tensor(cnot_singleq_error)

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(singleq_error, ["u"])
    noise_model.add_all_qubit_quantum_error(cnot_error, ["cx"])

    # cnot_error_rate = 2 * p_cnot - p_cnot ** 2
    cnot_error_rate = p_cnot

    two_qubit_error_map = np.zeros((n_qubits, n_qubits)) + cnot_error_rate

    return noise_model, two_qubit_error_map


#depolarizing_noise_model = create_depolarizing_error_model(p_cnot=0.01, p_u=0.001, p_meas=0.05)
