import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def plot_accuracy(algorithm, simulations, user_name):
    alg_path = os.path.join(f"results_{user_name}", algorithm)
    path = os.path.join(alg_path, "datacollector_sim")
    dfs = [pd.read_csv(path + str(sim) + ".csv") for sim in range(simulations)]
    size = dfs[0]["Accuracy"].size
    all_accuracies = []
    
    for variable in range(3):
        var_accuracy = [[] for _ in range(size)]
        for df in dfs:
            for obs in range(size):
                accuracy = df[["Accuracy"]].iloc[obs][0]
                acc_preproccesed = accuracy.strip("(),").split(",")
                number = float(acc_preproccesed[variable])
                var_accuracy[obs].append(number)
        all_accuracies.append(var_accuracy)

    ave_accuracies = [[], [], []]
    stds_accuracies = [[], [], []]
    for variable in range(3):
        for acc in all_accuracies[variable]:
            ave_accuracies[variable].append(np.mean(acc))
            stds_accuracies[variable].append(np.std(acc))

    variables = ["Accuracy", "F1", "Mathews Correlation Coeffcient"]
    for i in range(3):
        plt.figure()
        x = [0.01 * i for i in range(len(ave_accuracies[i]))]

        plt.plot(x, ave_accuracies[i], color="darkblue")
        if algorithm == "SBLN":
            plt.title("Mean accuracy over all tests for Levy based walk")
            plt.xlabel("iteration number (10^5)")

        elif algorithm == "GRAV":
            plt.title("Mean accuracy over all tests for gravitywalk")
            plt.xlabel("iteration number (10^5)")
        
        elif algorithm == "BM":
            plt.title("Mean accuracy over all tests for Brownian Motion")
            plt.xlabel("iteration number (10^6)")

        plt.ylabel(variables[i])

        plt.errorbar(x, ave_accuracies[i], yerr=stds_accuracies[i], alpha=0.1,
                    color="cornflowerblue")
        plt.savefig(os.path.join(alg_path, f"plot_{algorithm}_{variables[i]}.pdf"), dpi=300)
        plt.close()


def plot_network(road_dens, graph, user_name, gr_type):
    path = os.path.join(f"results_{user_name}", f"{gr_type}.pdf")
    width = road_dens.shape[1] - 1
    height = road_dens.shape[0] - 1

    pos = nx.get_node_attributes(graph, "pos")
    d = dict(graph.degree)
    nx.draw(graph, pos, node_size=[(v + 1) * 5 for v in d.values()])
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.title(f"{gr_type}")
    plt.savefig(path, dpi=300)
    plt.close()

def plot_networks(algorithm, simulations, config, user_name, threshold):

    alg_path = os.path.join(f"results_{user_name}", algorithm)
    path = os.path.join(alg_path, "rivalry_matrix_sim")
    matrices_sim = [np.load(path + str(sim) + ".npy") 
                    for sim in range(simulations)]

    width = config.road_dens.shape[1] - 1
    height = config.road_dens.shape[0] - 1

    shape = len(config.gang_info)
    for mat, matrix in enumerate(matrices_sim):
        graph = nx.Graph()
        for gang in config.gang_info.values():
            graph.add_node(gang.number, pos=gang.coords)

        for i in range(shape):
            total_interactions = matrix[i, :].sum()
            for j in range(shape):
                if total_interactions:
                    rival_strength = matrix[i][j] / total_interactions
                    if rival_strength > threshold:
                        graph.add_edge(i, j, color=config.colors[i])
        
        pos = nx.get_node_attributes(graph, "pos")
        d = dict(graph.degree)
        nx.draw(graph, pos, node_size=[(v + 1) * 5 for v in d.values()])
        if algorithm == "GRAV":
            plt.title("Network Gravity model")
        elif algorithm == "SBLN":
            plt.title("Network Semi-Biased Levy walk")
        elif algorithm == "BM":
            plt.title("Network Brownian Motion")
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.savefig(os.path.join(alg_path, f"network_sim{mat}.pdf"), dpi=300)
        plt.close()