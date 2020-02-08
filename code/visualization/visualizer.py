import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def plot_metrics(algorithm, simulations, user_name):
    alg_path = os.path.join(f"results_{user_name}", algorithm)
    path = os.path.join(alg_path, "datacollector_sim")
    dfs = [pd.read_csv(path + str(sim) + ".csv") for sim in range(simulations)]
    size = dfs[0]["Accuracy"].size
    all_accuracies, all_shape = [], []
    
    for variable in range(3):
        var_accuracy = [[] for _ in range(size)]
        var_shape = [[] for _ in range(size)]
        for df in dfs:
            for obs in range(size):
                accuracy = df[["Accuracy"]].iloc[obs][0]
                shapes = df[["Shape"]].iloc[obs][0]
                acc_preproccesed = accuracy.strip("(),").split(",")
                shapes_preproccesed = shapes.strip("(),").split(",")
                number = float(acc_preproccesed[variable])
                number2 = float(shapes_preproccesed[variable])
                var_accuracy[obs].append(number), var_shape[obs].append(number2)
        all_accuracies.append(var_accuracy), all_shape.append(var_shape)

    ave_accuracies, ave_shapes = [[], [], []], [[], [], []]
    stds_accuracies, stds_shapes = [[], [], []], [[], [], []]
    for variable in range(3):
        for acc, s in zip(all_accuracies[variable], all_shape[variable]):
            ave_accuracies[variable].append(np.mean(acc))
            ave_shapes[variable].append(np.mean(s))
            stds_accuracies[variable].append(np.std(acc))
            stds_shapes[variable].append(np.std(s))

    variables = ["Accuracy", "F1", "Mathews_Correlation_Coeffcient"]
    for i, acc in enumerate(ave_accuracies):
        plt.figure()
        x = [0.01 * i for i in range(len(acc))]

        plt.plot(x, acc, color="darkblue")
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

        plt.errorbar(x, acc, yerr=stds_accuracies[i], alpha=0.1,
                    color="cornflowerblue")
        plt.savefig(os.path.join(alg_path, f"plot_{algorithm}_{variables[i]}.pdf"), dpi=300)
        plt.close()

    variables = ["Density", "Variance_degree", "Centrality"]
    for i, shape_metric in enumerate(ave_shapes):
        plt.figure()
        x = [0.01 * i for i in range(len(shape_metric))]

        plt.plot(x, shape_metric, color="darkblue")
        if algorithm == "SBLN":
            plt.title(f"Mean {variables[i]} over all tests for Levy based walk")
            plt.xlabel("iteration number (10^5)")

        elif algorithm == "GRAV":
            plt.title(f"Mean {variables[i]} over all tests for gravitywalk")
            plt.xlabel("iteration number (10^5)")

        elif algorithm == "BM":
            plt.title(f"Mean {variables[i]} over all tests for Brownian Motion")
            plt.xlabel("iteration number (10^6)")

        plt.ylabel(variables[i])

        plt.errorbar(x, shape_metric, yerr=stds_shapes[i], alpha=0.1,
                     color="cornflowerblue")
        plt.savefig(os.path.join(
            alg_path, f"plot_{algorithm}_{variables[i]}.pdf"), dpi=300)
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
