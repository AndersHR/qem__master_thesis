import numpy as np
import matplotlib.pyplot as plt

from pgfplots_utilities import *

results_dir = "results/"

def plot_h2_UCCSD_distances_depolnoise():
    filename = "h2_UCCSD_vqe_distances_depolcnot.npz"

    file = np.load("results/" + filename)

    distances = file["distances"]

    energies = {}

    #energies["noisefree"] = file["energies_noisefree"]
    energies["exact"] = file["energies_exact"]
    energies["no_qem"] = file["energies_depolnoise"]
    energies["errordetect"] = file["energies_errordetect"]
    energies["zne"] = file["energies_depolnoise_zne"]
    energies["zne_and_errordetect"] = file["energies_depolnoise_zne_errordetect"]

    filename_exactenergies = "h2_distances_exactenergies.npz"

    file_exactenergies = np.load("results/" + filename_exactenergies)

    exact_energies = file_exactenergies["energies"]
    distances_exactenergies = file_exactenergies["distances"]

    # Write data to .dat file

    filename = "h2_UCCSD_vqe_distances_exact_energy.dat"
    column_names = ["dist", "exact"]
    data = np.asarray([distances_exactenergies, exact_energies])

    write_data_to_file(filename, data, column_names, fmt=["%f"]*len(data))

    filename = "h2_UCCSD_vqe_distances_depolcnot_energy.dat"
    column_names = ["dist", "no_qem", "errordetect", "zne", "zne_and_errordetect"]
    data = np.asarray([distances, energies["no_qem"], energies["errordetect"], energies["zne"],
                       energies["zne_and_errordetect"]])

    write_data_to_file(filename, data, column_names, fmt=["%f"]*len(data))

    # Plotting

    plt.title("VQE on H2, depolarizing error model w/ p_cnot=0.001")

    plt.ylabel(r"$E$, energy [H]")
    plt.xlabel(r"$a$, interatomic distance [Å]")

    plt.plot(distances_exactenergies, exact_energies, '-k', label="Exact")

    plt.plot(distances, energies["no_qem"],"+", label="No qem")
    plt.plot(distances, energies["errordetect"],"x", label="Error detection")
    plt.plot(distances, energies["zne"],"^", label="ZNE")
    plt.plot(distances, energies["zne_and_errordetect"],"o", label="ZNE and error detection")

    plt.legend()

    plt.show()

def plot_h2_UCCSD_distances_depolnoise_deviation():
    filename = "h2_UCCSD_vqe_distances_depolcnot.npz"

    file = np.load("results/" + filename)

    distances = file["distances"]

    energies = {}

    #energies["noisefree"] = file["energies_noisefree"]
    energies["exact"] = file["energies_exact"]
    energies["no_qem"] = file["energies_depolnoise"]
    energies["errordetect"] = file["energies_errordetect"]
    energies["zne"] = file["energies_depolnoise_zne"]
    energies["zne_and_errordetect"] = file["energies_depolnoise_zne_errordetect"]

    deviation = {}

    deviation["no_qem"] = np.abs(energies["exact"] - energies["no_qem"])
    deviation["errordetect"] = np.abs(energies["exact"] - energies["errordetect"])
    deviation["zne"] = np.abs(energies["exact"] - energies["zne"])
    deviation["zne_and_errordetect"] = np.abs(energies["exact"] - energies["zne_and_errordetect"])

    chemical_accuracy = np.zeros(np.shape(distances)[0]) + 1.6*(10**-3)

    # Write data to .dat file

    filename = "h2_UCCSD_vqe_distances_depolcnot_deviation.dat"
    column_names = ["dist", "no_qem", "errordetect", "zne", "zne_and_errordetect"]
    data = np.asarray([distances, deviation["no_qem"], deviation["errordetect"], deviation["zne"],
                       deviation["zne_and_errordetect"]])

    write_data_to_file(filename, data, column_names)

    # Plotting

    plt.title("VQE on H2, depolarizing error model w/ p_cnot=0.001")

    plt.ylim(0, 0.18)

    plt.ylabel(r"$\Delta$, deviation from exact energy [H]")
    #plt.ylabel(r"$\Delta$, Error [H]", fontsize=15)
    plt.xlabel(r"$a$, interatomic distance [Å]")

    plt.plot(distances, chemical_accuracy, '--', label="Chemical accuracy")

    plt.plot(distances, deviation["no_qem"],"+", label="No qem")
    plt.plot(distances, deviation["errordetect"],"x", label="Error detection")
    plt.plot(distances, deviation["zne"],"^", label="ZNE")
    plt.plot(distances, deviation["zne_and_errordetect"],"o", label="ZNE and error detection")

    plt.legend()

    plt.show()


def plot_h2_UCCSD_errorrates_depolnoise():
    filename = "h2_UCCSD_vqe_errorrates_depolcnot.npz"

    file = np.load("results/" + filename)

    error_rates = file["error_rates"]

    energies, discarded = {}, {}

    energies["exact"] = file["energies_exact"]
    energies["no_qem"] = file["energies_depolnoise"]
    energies["errordetect"] = file["energies_errordetect"]
    energies["zne"] = file["energies_depolnoise_zne"]
    energies["zne_and_errordetect"] = file["energies_depolnoise_zne_errordetect"]

    print("DISCARDED")
    discarded["zne_and_errordetect"] = file["discarded_zne_errordetect"]
    print(discarded["zne_and_errordetect"])

    deviation = {}

    deviation["no_qem"] = np.abs(energies["exact"] - energies["no_qem"])
    deviation["errordetect"] = np.abs(energies["exact"] - energies["errordetect"])
    deviation["zne"] = np.abs(energies["exact"] - energies["zne"])
    deviation["zne_and_errordetect"] = np.abs(energies["exact"] - energies["zne_and_errordetect"])

    relative_deviation = {}

    relative_deviation["no_qem"] = np.abs((energies["no_qem"] - energies["exact"])/energies["exact"])
    relative_deviation["errordetect"] = np.abs((energies["errordetect"] - energies["exact"]) / energies["exact"])
    relative_deviation["zne"] = np.abs((energies["zne"] - energies["exact"]) / energies["exact"])
    relative_deviation["zne_and_errordetect"] = np.abs((energies["zne_and_errordetect"] - energies["exact"]) / energies["exact"])

    print(energies["exact"])
    print(energies["zne_and_errordetect"])
    print(energies["exact"] - energies["zne_and_errordetect"])
    print(deviation["zne_and_errordetect"])
    print(relative_deviation["zne_and_errordetect"])

    # Save data to .dat file for pgf plots

    filename = "h2_UCCSD_vqe_errorrates_depolcnot_deviation.dat"
    column_names = ["error_rates", "error_rates_percent", "no_qem", "errordetect", "zne", "zne_and_errordetect"]
    data = np.asarray([error_rates, 100*np.asarray(error_rates), deviation["no_qem"], deviation["errordetect"], deviation["zne"],
                       deviation["zne_and_errordetect"]])

    write_data_to_file(filename, data, column_names)

    filename = "h2_UCCSD_vqe_errorrates_depolcnot_relativedeviation.dat"
    column_names = ["error_rates", "error_rates_percent", "no_qem", "errordetect", "zne", "zne_and_errordetect"]
    data = np.asarray(
        [error_rates, 100 * np.asarray(error_rates), relative_deviation["no_qem"], relative_deviation["errordetect"],
         relative_deviation["zne"], relative_deviation["zne_and_errordetect"]])

    write_data_to_file(filename, data, column_names)

    # Plotting

    plt.title("H2 VQE at a=0.74 Å, depolarizing error models")

    plt.ylim(0, 0.92)

    plt.ylabel(r"$\Delta$, deviation from exact energy [H]")
    plt.xlabel(r"$p$, CNOT error rate")
    plt.xticks([0.001, 0.005, 0.01, 0.015, 0.02], ["0.1%","0.5%", "1%", "1.5%", "2%",])

    plt.plot(error_rates, deviation["no_qem"],"+", label="No qem")
    plt.plot(error_rates, deviation["errordetect"],"x", label="Error detection")
    plt.plot(error_rates, deviation["zne"],"^", label="ZNE")
    plt.plot(error_rates, deviation["zne_and_errordetect"],"o", label="ZNE and error detection")

    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_h2_UCCSD_distances_depolnoise()

    plot_h2_UCCSD_distances_depolnoise_deviation()

    plot_h2_UCCSD_errorrates_depolnoise()