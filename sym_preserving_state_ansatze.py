from qiskit import *
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info.operators import Operator

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

import numpy as np
from typing import *

# [1] Efficient Symmetry-Preserving State Preparation Circuits for the VQE Algorithm: https://arxiv.org/pdf/1904.10910.pdf

def apply_r_gate(qc: QuantumCircuit, q: int, theta: float, phi: float):
    qc.ry(theta + np.pi / 2, q)
    qc.rz(phi + np.pi, q)


def apply_rdag_gate(qc: QuantumCircuit, q: int, theta: Union[float, Parameter], phi: Union[float, Parameter]):
    qc.rz(-(phi + np.pi), q)
    qc.ry(-(theta + np.pi / 2), q)


def apply_r_gate_alt(qc: QuantumCircuit, q: int, theta: float, phi: float):
    qc.rz(phi + np.pi, q)
    qc.ry(theta + np.pi / 2, q)


"""
def apply_a_gate(qc: QuantumCircuit, q1: int, q2: int, theta: float, phi: float):
    a_operator = Operator([[1,0,0,0],
                           [0,np.cos(theta),np.exp(1j*phi)*np.sin(theta),0],
                           [0,np.exp(-1j*phi)*np.sin(theta),-np.cos(theta),0],
                           [0,0,0,1]])
    qc.unitary(a_operator, [q1,q2], label="A({:.2f},{:.2f})".format(theta, phi))
"""

def apply_a_gate(qc: QuantumCircuit, q1: int, q2: int, theta: Union[float, Parameter], phi: Union[float, Parameter]):
    qc.cx(q1, q2)
    apply_rdag_gate(qc, q1, theta, phi)
    qc.cx(q2, q1)
    apply_r_gate(qc, q1, theta, phi)
    qc.cx(q1, q2)

def apply_iterative_a_gate(qc: QuantumCircuit, q1: int, q2: int, i: int, params: ParameterVector):
    apply_a_gate(qc, q1, q2, params[i], params[i+1])
    return i+2

def apply_a_gate_column(qc, q_start, q_end):
    return

def get_n4_m2_particlepreserving_ansatz(params: Union[list, np.ndarray], n_qubits: int = 5,
                                        noise_amplification_factor: Optional[int] = 1,
                                        error_detection: Optional[bool] = False,
                                        unroll: Optional[bool] = True, version: int = 1):

    qc = QuantumCircuit(n_qubits, n_qubits)

    if version == 1:

        apply_r_gate(qc, 0, params[0], params[1])

        for j in range(noise_amplification_factor):
            qc.cx(0, 1)

        apply_r_gate(qc, 0, params[2], params[3])
        apply_r_gate(qc, 1, params[4], params[5])

        for j in range(noise_amplification_factor):
            qc.cx(0, 2)
            qc.cx(1, 3)

        qc.x(2)
        qc.x(3)

        apply_a_gate(qc, 0, 1, params[6], params[7])
        apply_a_gate(qc, 2, 3, params[8], params[9])

        if error_detection:
            for i in range(3):
                qc.cx(i, 4)

    elif version == 2:

        qc.x(1)
        qc.x(2)

        apply_a_gate(qc, 0, 1, params[1], params[0])
        apply_a_gate(qc, 2, 3, params[2], params[0])

        apply_a_gate(qc, 1, 2, params[3], params[0])

        apply_a_gate(qc, 0, 1, params[4], params[5])
        apply_a_gate(qc, 2, 3, params[6], params[7])

        apply_a_gate(qc, 1, 2, params[8], params[9])

    if unroll:
        unroller_pass = Unroller(["u3", "cx"])
        pm = PassManager(unroller_pass)

        qc = pm.run(qc)

    return qc


def get_n4_m2_parameterized_ansatz(param_name: str = "p", tot_qubits: int = 5, tot_clbits: int = 5)\
        -> (QuantumCircuit, ParameterVector):
    params = ParameterVector(param_name, 10)
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    apply_r_gate(qc, 0, params[0], params[1])

    qc.cx(0, 1)

    apply_r_gate(qc, 0, params[2], params[3])
    apply_r_gate(qc, 1, params[4], params[5])

    qc.cx(0,2)
    qc.cx(1,3)
    qc.x([0,1])

    apply_a_gate(qc, 0, 1, params[6], params[7])
    apply_a_gate(qc, 2, 3, params[8], params[9])

    return qc, params

def get_n4_m2_errordetection_circuit():
    qc = QuantumCircuit(6,6)

    qc.cx(0,4)
    qc.cx(1,4)
    qc.cx(2,5)
    qc.cx(3,5)

    qc.measure([4,5],[4,5])

    return qc, [4,5]

def n4_m2_errordetection_decision_rule(mmt_str):
    if mmt_str == "01" or mmt_str == "10":
        return True
    else:
        return False

def get_errordetecqc_n4_m2_particlenum_4ancillas(tot_qubits: int = 8, tot_clbits: int = 8):
    qc = QuantumCircuit(tot_qubits, tot_clbits)
    error_detect_qubits = [4,5,6,7]

    qc.cx(0,4)
    qc.cx(1,5)
    qc.cx(2,6)
    qc.cx(3,7)

    qc.measure(error_detect_qubits, error_detect_qubits)

    return qc, error_detect_qubits

def decision_rule_n4_m2_particlenum_4ancilllas(bit_str: str):
    num_ones = bit_str.count("1")
    return num_ones != 2

def get_n12_m2_parametrized_ansatz(param_name: str = "p", tot_qubits: int = 12, tot_clbits: int = 12):
    if tot_qubits < 12:
        raise Exception("tot_qubits must be more than 12")
    if tot_clbits < 12:
        raise Exception("tot_clbits must be more than 12")

    num_params = 2*66 - 2

    params = ParameterVector(param_name, num_params)
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    qc.x(3)
    qc.x(8)

    i = 0

    # First layer (2 gates)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)

    # Second layer (4 gates)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)

    # Third layer (5 gates)
    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 5, 6, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)
    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    # Fourth layer (6 gates)
    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    # SO FAR: 17 GATES.
    #print("17 gates", i)

    #
    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 5, 6, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)
    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    #
    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    # SO FAR: 28 GATES
    #print("28 gates", i)

    #
    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 5, 6, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)
    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    #
    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    # SO FAR: 39 GATES
    #print("39 gates:", i)

    #
    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 5, 6, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)
    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    #
    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    # SO FAR: 50 GATES
    #print("50 gates:", i)

    #
    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    i = apply_iterative_a_gate(qc, 3, 4, i, params)
    i = apply_iterative_a_gate(qc, 5, 6, i, params)
    i = apply_iterative_a_gate(qc, 7, 8, i, params)
    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    #
    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    # SO FAR: 61 GATES
    #print("61 gates:", i)

    i = apply_iterative_a_gate(qc, 1, 2, i, params)
    apply_a_gate(qc, 3, 4, params[i], params[i+1])
    apply_a_gate(qc, 5, 6, params[i], params[i+2])
    apply_a_gate(qc, 7, 8, params[i], params[i+3])
    i = i+4

    i = apply_iterative_a_gate(qc, 9, 10, i, params)

    #print("PARAMETERS USED: {:}".format(i))

    return qc, params

def get_n12_m4_parametrized_ansatz(param_name: str = "p", tot_qubits: int = 12, tot_clbits: int = 12):
    if tot_qubits < 12:
        raise Exception("tot_qubits must be more than 12")
    if tot_clbits < 12:
        raise Exception("tot_clbits must be more than 12")

    num_params = 2*495 - 2

    params = ParameterVector(param_name, num_params)
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    qc.x(2)
    qc.x(4)
    qc.x(7)
    qc.x(9)

    apply_a_gate(qc, 1, 2, params[0], params[1])
    apply_a_gate(qc, 3, 4, params[0], params[2])
    apply_a_gate(qc, 7, 8, params[0], params[3])
    apply_a_gate(qc, 9, 10, params[4], params[5])

    i = 6

    i = apply_iterative_a_gate(qc, 0, 1, i, params)
    i = apply_iterative_a_gate(qc, 2, 3, i, params)
    i = apply_iterative_a_gate(qc, 4, 5, i, params)
    i = apply_iterative_a_gate(qc, 6, 7, i, params)
    i = apply_iterative_a_gate(qc, 8, 9, i, params)
    i = apply_iterative_a_gate(qc, 10, 11, i, params)

    for j in range(44):
        i = apply_iterative_a_gate(qc, 1, 2, i, params)
        i = apply_iterative_a_gate(qc, 3, 4, i, params)
        i = apply_iterative_a_gate(qc, 5, 6, i, params)
        i = apply_iterative_a_gate(qc, 7, 8, i, params)
        i = apply_iterative_a_gate(qc, 9, 10, i, params)

        i = apply_iterative_a_gate(qc, 0, 1, i, params)
        i = apply_iterative_a_gate(qc, 2, 3, i, params)
        i = apply_iterative_a_gate(qc, 4, 5, i, params)
        i = apply_iterative_a_gate(qc, 6, 7, i, params)
        i = apply_iterative_a_gate(qc, 8, 9, i, params)
        i = apply_iterative_a_gate(qc, 10, 11, i, params)

    i = apply_iterative_a_gate(qc, 5, 6, i, params)

    print(i)

    return qc, params


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #print(decision_rule_n4_m2_particlenum_4ancilllas("1111"))

    qc, params = get_n12_m4_parametrized_ansatz()

    print(qc)
    print(params)
    print(len(params))
