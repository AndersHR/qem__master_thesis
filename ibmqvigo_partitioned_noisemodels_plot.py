import matplotlib.pyplot as plt
import numpy as np

import os, sys, pickle

if __name__ == "__main__":
    N_AMP_FACTORS = 7

    filename = "results/ibmqlondon_partitioned_noisemodels_results"

    file = open(filename, "rb")
    experiments = pickle.load(file)
    file.close()

    print(experiments)

    amp_factors = np.asarray([i+1 for i in range(N_AMP_FACTORS)])

    column_names = ["amp_factors"]
    data_mitigated = [amp_factors.copy()]
    data_noise_amplified = [amp_factors.copy()]

    plt.plot(amp_factors, [0.5 for i in range(N_AMP_FACTORS)], "--", label="ideal")

    for name in experiments.keys():
        column_names.append(name)
        data_mitigated.append(experiments[name]["mitigated_exp_vals"][0:N_AMP_FACTORS])
        data_noise_amplified.append(experiments[name]["noise_amplified_exp_vals"][0:N_AMP_FACTORS])

        plt.plot(amp_factors, experiments[name]["mitigated_exp_vals"][0:N_AMP_FACTORS], "o-", label=name)

    plt.legend()
    plt.show()
