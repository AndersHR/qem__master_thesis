
def construct_pauli_noise_model(p_cnot: float = 0.01, p_single_q: float = 0.001, p_meas: float = 0.05,
                                single_qubit_noise: bool = True, measurement_noise: bool = True):
    noise_model = NoiseModel()

    # CNOT noise
    cnot_bitflip_error = pauli_error([("X", p_cnot), ("I", 1-p_cnot)])
    cnot_phaseflip_error = pauli_error([("Z", p_cnot), ("I", 1-p_cnot)])

    cnot_pauli_error_composed = cnot_bitflip_error.compose(cnot_phaseflip_error)
    cnot_pauli_error = cnot_pauli_error_composed.tensor(cnot_pauli_error_composed)

    noise_model.add_all_qubit_quantum_error(cnot_pauli_error, ["cx"])

    # Single-qubit noise
    if single_qubit_noise:
        single_q_bitflip_error = pauli_error([("X", p_single_q), ("I", 1-p_single_q)])
        single_q_phaseflip_error = pauli_error([("Z", p_single_q), ("I", 1-p_single_q)])

        single_q_pauli_error = single_q_bitflip_error.compose(single_q_phaseflip_error)

        noise_model.add_all_qubit_quantum_error(single_q_pauli_error, ["u2", "u3"])

    # Measurement noise
    if measurement_noise:
        meas_bitflip_error = pauli_error([("X", p_meas), ("I", 1-p_meas)])
        meas_phaseflip_error = pauli_error([("Z", p_meas), ("I", 1-p_meas)])

        meas_pauli_error = meas_bitflip_error.compose(meas_phaseflip_error)

        noise_model.add_all_qubit_quantum_error(meas_pauli_error, ["measure"])

    return noise_model

def construct_depol_noise_model(p: float):
    if p > 1.0:
        raise Exception("Invalid probability, must be p <= 1.0")

    X = np.asarray([[0,1],[1,0]])
    Y = np.asarray([[0, -1j],[1j,0]])
    Z = np.asarray([[1,0],[0,-1]])
    I = np.asarray([[1,0],[0,1]])

    kraus_operators = [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]

    error = QuantumError(noise_ops=kraus_operators)
    cx_error = error.tensor(error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(cx_error, ["cx"])

    return noise_model

def construct_cnot_depol_noise_model(p: float = 0.001):
    if p > 1.0:
        raise Exception("Invalid probability, must be p <= 1.0")

    X = np.asarray([[0, 1], [1, 0]])
    Y = np.asarray([[0, -1j], [1j, 0]])
    Z = np.asarray([[1, 0], [0, -1]])
    I = np.asarray([[1, 0], [0, 1]])

    pauli_dict = {"X": X, "Y": Y, "Z": Z, "I": I}

    kraus_operators = [np.sqrt(1-p)*np.kron(I,I)]

    for a in ["I","X","Y","Z"]:
        for b in ["I","X","Y","Z"]:
            if not ((a=="I") and (b=="I")):
                op = np.kron(pauli_dict[a], pauli_dict[b])
                kraus_operators.append(np.sqrt(p/15)*op)

    error = QuantumError(noise_ops=kraus_operators, number_of_qubits=2)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ["cx"])

    return noise_model
