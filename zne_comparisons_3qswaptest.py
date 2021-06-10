from qiskit import *

from qiskit.test.mock import FakeAthens #FakeBelem
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet

from qiskit.providers.aer import Aer

import numpy as np
import os, sys, pickle
from dataclasses import dataclass

from typing import *

sys.path.append("../")

from error_mitigation.zero_noise_extrapolation import *
from error_mitigation.zero_noise_extrapolation_cnot import *
from qiskit_utilities.utilities import *

from swaptest_circuit import qc_swaptest, swaptest_exp_val_func, swaptest_exp_val_func_randompauli
from noisemodels import *
from qem_master_thesis__chemistry.pgfplots_utilities import *

DIRECTORY_RESULTS = "results"
DIRECTORY_DATA = "data"

EXPERIMENT_NAME_BASE = "zne_comparisons_3qswaptest"

FILENAME_BASE = os.path.dirname(__file__) + "/" + DIRECTORY_RESULTS + "/" + EXPERIMENT_NAME_BASE

SHOTS_PER = 8192
REPEATS = 1024
SHOTS = REPEATS * SHOTS_PER
N_AMP_FACTORS = 7

#IBMQ.save_account("Your IBMQ token")
IBMQ.load_account()

provider = IBMQ.get_provider()

PHYS_BACKEND = provider.get_backend("ibmq_athens")

PHYS_BACKEND_2 = provider.get_backend("ibmq_belem")

SIM_BACKEND = Aer.get_backend("qasm_simulator")
MOCK_BACKEND = FakeAthens()
#MOCK_BACKEND_2 = FakeBelem()

def transpile_circuit(qc: QuantumCircuit, backend):
    filename = FILENAME_BASE + "_{:}".format(backend.name()) + ".circuit"

    if os.path.isfile(filename):
        file = open(filename, "rb")
        qc_transpiled = pickle.load(file)
        file.close()
        print("Transpiled circuit loaded")
    else:
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        file = open(filename, "wb")
        pickle.dump(qc_transpiled, file)
        file.close()
        print("Transpiled circuit created and saved")

    return qc_transpiled

def compute_repeatingcnots(qc, backend, noise_model=None, experiment_name=""):
    global SHOTS, N_AMP_FACTORS

    experiment_name = EXPERIMENT_NAME_BASE + experiment_name

    qem = ZeroNoiseExtrapolation(qc=qc, exp_val_func=swaptest_exp_val_func, backend=backend,
                                 noise_model=noise_model, n_amp_factors=N_AMP_FACTORS, shots=SHOTS,
                                 save_results=True, experiment_name=experiment_name)

    qem.mitigate(verbose=True)

    noise_amplified_exp_vals = qem.noise_amplified_exp_vals
    mitigated_exp_vals = np.zeros(N_AMP_FACTORS)

    mitigated_exp_vals[0] = noise_amplified_exp_vals[0]

    #print("Amp factor=",1)
    #print("depth:", qem.qc.depth())
    #print("op_count:", qem.qc.count_ops())


    for i in range(1, N_AMP_FACTORS):
        mitigated_exp_vals[i] = Richardson_extrapolate(noise_amplified_exp_vals[0:(i+1)],
                                                       qem.noise_amplification_factors[0:(i+1)])[0]

        qc_amplified = qem.noise_amplify_and_pauli_twirl_cnots(qc=qem.qc, amp_factor=(2*i + 1),
                                                               pauli_twirl=False)
        print("Amp factor=",2*i+1)
        print("depth:", qc_amplified.depth())
        print("op_count:", qc_amplified.count_ops())

    amp_factors = qem.noise_amplification_factors

    variances = 1 - noise_amplified_exp_vals**2

    column_names = np.asarray(["n",
                               "amp_factors",
                               "noise_amplified_exp_vals",
                               "mitigated_exp_vals",
                               "ideal",
                               "absolute_error",
                               "relative_error",
                               "variances"])

    data = np.asarray([[i + 1 for i in range(N_AMP_FACTORS)],
                       amp_factors,
                       noise_amplified_exp_vals,
                       mitigated_exp_vals,
                       [0.5 for i in range(N_AMP_FACTORS)],
                       np.abs(mitigated_exp_vals - 0.5),
                       np.abs(mitigated_exp_vals - 0.5) / 0.5,
                       variances
                       ])

    write_data_to_file(filename=experiment_name + "_cnotrep.dat", data=data, column_names=column_names,
                       directory=DIRECTORY_DATA)

    return noise_amplified_exp_vals, mitigated_exp_vals

def compute_randompauli_qem(qc, backend, execution_backend=None, two_qubit_error_map=None, noise_model=None, experiment_name="", paulitwirling=False):
    global SHOTS_PER, REPEATS, N_AMP_FACTORS, EXPERIMENT_NAME_BASE

    experiment_name = EXPERIMENT_NAME_BASE + experiment_name

    if paulitwirling:
        experiment_name += "_paulitwirling"

    if noise_model is None and two_qubit_error_map is None:
        two_qubit_error_map = get_cx_error_map(backend)

    #if noise_model is None:
    #    noise_model = NoiseModel.from_backend(backend)

    if execution_backend is None:
        execution_backend = SIM_BACKEND

    basis_gates = ["cx", "id", "u3", "u2", "x"]

    amplification_factors_powersoftwo = [2 ** i for i in range(N_AMP_FACTORS)]

    for i in range(2, N_AMP_FACTORS + 1):
        amp_factors_partitioned = amplification_factors_powersoftwo[0:i]

        R, E_dict, E_av_dict, \
        max_depth_dict, mean_depth_dict, \
        max_depth_transpiled_dict, mean_depth_transpiled_dict, \
        bn = mitigate(qc, amp_factors_partitioned,
                      swaptest_exp_val_func_randompauli,
                      execution_backend,
                      experiment_name, two_qubit_error_map,
                      SHOTS_PER, REPEATS,
                      backend,
                      noise_model,
                      basis_gates,
                      paulitwirling=paulitwirling
                      )

    noise_amplified_exp_vals = np.zeros(N_AMP_FACTORS)
    mitigated_exp_vals = np.zeros(N_AMP_FACTORS)
    for i, r in enumerate(amplification_factors_powersoftwo):
        noise_amplified_exp_vals[i] = E_av_dict[bn + "_r{:}".format(r)][-1]

    mitigated_exp_vals[0] = noise_amplified_exp_vals[0]

    for i in range(1, N_AMP_FACTORS):
        mitigated_exp_vals[i] = Richardson_extrapolate(noise_amplified_exp_vals[0:(i+1)],
                                                       np.asarray(amplification_factors_powersoftwo[0:(i+1)]))[0]

    column_names = np.asarray(["n",
                               "amp_factors",
                               "noise_amplified_exp_vals",
                               "mitigated_exp_vals",
                               "ideal"])
    data = np.asarray([[i+1 for i in range(N_AMP_FACTORS)],
                       amplification_factors_powersoftwo,
                       noise_amplified_exp_vals,
                       mitigated_exp_vals,
                       [0.5 for i in range(N_AMP_FACTORS)]])

    write_data_to_file(filename=experiment_name + "_randompauli.dat", data=data, column_names=column_names,
                       directory=DIRECTORY_DATA)

    return noise_amplified_exp_vals, mitigated_exp_vals

def compute_randompauli_qem_paulitwirling(qc, backend, execution_backend=None, two_qubit_error_map=None, noise_model=None, experiment_name=""):
    return compute_randompauli_qem(qc=qc, backend=backend, execution_backend=execution_backend, two_qubit_error_map=two_qubit_error_map, noise_model=noise_model,
                                   experiment_name=experiment_name, paulitwirling=True)

def compute_noisemodel_amplification(qc, noise_model_func: Callable, amp_factor_mode: str = "odd",
                                     meas_noise: bool = False, experiment_name: str = ""):
    p_u = 0.001
    p_cnot = 0.01
    p_meas = 0

    if amp_factor_mode == "odd":
        noise_amplification_factors = [2*i + 1 for i in range(N_AMP_FACTORS)]
    elif amp_factor_mode == "square":
        noise_amplification_factors = [2**i for i in range(N_AMP_FACTORS)]
    else:
        raise Exception("Did not recognise amp_factor_mode={:}".format(amp_factor_mode))

    noise_models = []

    for amp_factor in noise_amplification_factors:
        if meas_noise == True:
            nm = noise_model_func(p_cnot=p_cnot, p_u=p_u, p_meas=p_meas)
        else:
            nm = noise_model_func(p_cnot=p_cnot, p_u=p_u)
        noise_models.append(nm)

    noise_amplified_exp_vals = np.empty(N_AMP_FACTORS)

    for amp_factor in noise_amplification_factors:
        filename = "results/" + experiment_name
        filename += "_noisemodelamplification_{:}ampfactors_r{:}.results".format(amp_factor_mode, amp_factor)

        if os.path.isfile(filename):
            file = open(filename, "rb")
            exp_val = pickle.load(file)
            file.close()
        else:
            circuits = [qc.copy() for i in range(REPEATS)]
            job = execute(circuits, backend=SIM_BACKEND, shots=SHOTS_PER, optimization_level=0)
            results = job.result().results

            exp_val, _ = swaptest_exp_val_func(results)

            file = open(filename, "wb")
            pickle.dump(exp_val, file)
            file.close()







# EXPERIMENTS:

def mockbackend_ibmqathens():
    experiment_name = "_mockbackend_{:}".format(MOCK_BACKEND)

    print(">>> NOISE MODEL from backend {:} <<<".format(MOCK_BACKEND))

    qc = transpile_circuit(qc=qc_swaptest, backend=MOCK_BACKEND)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc, backend=MOCK_BACKEND,
                                                                          experiment_name=experiment_name)
    print(mitigated_exp_vals)

    #noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=PHYS_BACKEND,
    #                                                                       execution_backend=MOCK_BACKEND,
    #                                                                       experiment_name=experiment_name)
    #
    #print(mitigated_exp_vals)

def ibmqdevice_noisemodel_ibmqathens():
    experiment_name = "_nmfrombackend_{:}".format(PHYS_BACKEND)

    print(">>> NOISE MODEL from backend {:} <<<".format(PHYS_BACKEND))

    qc = transpile_circuit(qc=qc_swaptest, backend=PHYS_BACKEND)

    noise_model = NoiseModel.from_backend(PHYS_BACKEND)
    two_qubit_error_map = get_cx_error_map(PHYS_BACKEND)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc, backend=SIM_BACKEND,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name)

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=PHYS_BACKEND,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=PHYS_BACKEND,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)

    print(mitigated_exp_vals)


def ibmqdevice_noisemodel_ibmqbelem():
    experiment_name = "_nmfrombackend_{:}".format(PHYS_BACKEND_2)

    print(">>> NOISE MODEL from backend {:} <<<".format(PHYS_BACKEND_2))

    qc = transpile_circuit(qc=qc_swaptest, backend=PHYS_BACKEND_2)

    noise_model = NoiseModel.from_backend(PHYS_BACKEND_2)
    two_qubit_error_map = get_cx_error_map(PHYS_BACKEND_2)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc, backend=SIM_BACKEND,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name)

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=PHYS_BACKEND_2,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=PHYS_BACKEND_2,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)

    print(mitigated_exp_vals)


def depolarizing_noisemodel():
    experiment_name = "_depolnm"

    print(">>> DEPOLARIZING NOISE MODEL <<<")

    qc = transpile_circuit(qc=qc_swaptest, backend=SIM_BACKEND)

    p_cnot, p_u, p_meas = 0.01, 0.001, 0#0.05

    backend = SIM_BACKEND
    noise_model, two_qubit_error_map = create_depolarizing_error_model(p_cnot, p_u, p_meas)

    print(two_qubit_error_map)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc,
                                                                          backend=backend,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name,
                                                                          )

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)


def amplitudedamping_noisemodel():
    experiment_name = "_amplitudedampingnm"

    print(">>> AMPLITUDE DAMPING NOISE MODEL <<<")

    qc = transpile_circuit(qc=qc_swaptest, backend=SIM_BACKEND)

    p_cnot, p_u, p_meas = 0.01, 0.001, 0.05

    backend = SIM_BACKEND
    noise_model, two_qubit_error_map = create_amplitude_damping_error_model(p_cnot=p_cnot, p_u=p_u)

    print(two_qubit_error_map)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc,
                                                                          backend=backend,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name,
                                                                          )

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)


def phasedamping_noisemodel():
    experiment_name = "_phasedampingnm"

    print(">>> PHASE DAMPING NOISE MODEL <<<")

    qc = transpile_circuit(qc=qc_swaptest, backend=SIM_BACKEND)

    p_cnot, p_u, p_meas = 0.01, 0.001, 0.05

    backend = SIM_BACKEND
    noise_model, two_qubit_error_map = create_phase_damping_error_model(p_cnot=p_cnot, p_u=p_u)

    print(two_qubit_error_map)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc,
                                                                          backend=backend,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name,
                                                                          )

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)

def paulibitphaseflip_noisemodel():
    experiment_name = "_bitphaseflipnm"

    print(">>> AMPLITUDE DAMPING NOISE MODEL <<<")

    qc = transpile_circuit(qc=qc_swaptest, backend=SIM_BACKEND)

    p_cnot, p_u, p_meas = 0.01, 0.001, 0.05

    backend = SIM_BACKEND
    noise_model, two_qubit_error_map = create_bit_phase_flip_error_model(p_cnot=p_cnot, p_u=p_u)

    print(two_qubit_error_map)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_repeatingcnots(qc=qc,
                                                                          backend=backend,
                                                                          noise_model=noise_model,
                                                                          experiment_name=experiment_name,
                                                                          )

    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)

    noise_amplified_exp_vals, mitigated_exp_vals = compute_randompauli_qem_paulitwirling(qc=qc, backend=backend,
                                                                           noise_model=noise_model,
                                                                           two_qubit_error_map=two_qubit_error_map,
                                                                           experiment_name=experiment_name)
    print(mitigated_exp_vals)




def physical_ibmqdevice():
    """
    Runs of the 3qubit SWAP-test circuit from physical ibmq device. The runs are sent to IBMQ through a IBMQ.JobManager
    from a python notebook (ibmq_physicaldevice_experiments.ipynb). After runnning, we retrieve the results from IBMQ
    and process them here.

    The job_set instances with the corresponding jobs are retrieved through a set of job_set ids.
    These are hard-coded in here.

    :return:
    """
    job_sets = {}

    noise_amplification_factors = np.asarray([2*i+1 for i in range(N_AMP_FACTORS)])

    for i, amp_factor in enumerate(noise_amplification_factors):
        job_sets["r{:}".format(amp_factor)] = {}

    job_sets["r1"]["ids"] = ["4b1dd6bed0ee4fd2a5fd549870f6632b-1618388485079264",
                             "4cfb98cab90941e281924ae78572f7b7-16183986319119437",
                             "42639d51fec64cabb43998a4f784dbfb-1618404688467747",
                             "d305960a874e441186707361f7516863-16184080281586678",
                             ]
    job_sets["r3"]["ids"] = ["8c7b4c31f44e44c88ccaef3e8272728e-16184148284139018",
                             ]
    job_sets["r5"]["ids"] = ["d300582e363949b6a6a60c2a472e0aaf-1618472943752947",
                             "cf6538bcc46c4d64b115b3dec0571a7f-16192586847221723"
                             ]
    job_sets["r7"]["ids"] = ["1155ad63d1794802b350c3b9a79edb84-16184751399301562",
                             "9dbcce80c89946a9931c14c1f431646b-1618485353539269"
                             ]
    job_sets["r9"]["ids"] = ["a2106e75601a4b0dba4f6737784cfd1c-1618489913209879", "98dee5e4713d4c13a7e2dbfacd6b702c-16192642422102523"]
    job_sets["r11"]["ids"] = ["e4655546e9e14220b62c984ab2c797a6-1618496800783153"]
    job_sets["r13"]["ids"] = ["1dda94358e9e469598cba42366cc73a5-16184999242610683",
                              "6b80e912b9e74d1c9ecb2fb833bb68c1-16185617700790067",
                              "793a3104f98f4f989030b6f680b85032-16185815837021818"]

    for key in job_sets.keys():
        job_manager = IBMQJobManager()
        job_sets[key]["job_sets"] = []
        job_sets[key]["jobs"] = []
        for job_set_id in job_sets[key]["ids"]:
            print(key, "-" ,job_set_id)
            try:
                job_set = job_manager.retrieve_job_set(job_set_id=job_set_id, provider=provider)
                jobs = job_set.jobs()
                job_sets[key]["job_sets"].append(job_set)
                for job in jobs:
                    if job is not None:
                        job_sets[key]["jobs"].append(job)
                print("successfully retrieved job set")
            except:
                print("Error for {:} job set id={:}".format(key, job_set_id))

    print(job_sets)


if __name__ == "__main__":
    # compute(SIM_BACKEND)

    #mockbackend_ibmqathens()

    ibmqdevice_noisemodel_ibmqathens()

    ibmqdevice_noisemodel_ibmqbelem()

    depolarizing_noisemodel()

    amplitudedamping_noisemodel()

    phasedamping_noisemodel()

    #paulibitphaseflip_noisemodel()

    #physical_ibmqdevice()
