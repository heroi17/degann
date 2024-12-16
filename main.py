from os import read
from degann.search_algorithms import genetic_search, grid_search, simulated_annealing, random_search
import numpy as np
from experiments.functions import gauss
from random import randint
import matplotlib.pyplot as plt
import time
from degann.networks.imodel import *
from itertools import product

def plotGraph(predicted_data, given_data) -> None:
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 1, 3, 5])
    
    # 1. Создание графика
    plt.figure(figsize=(8, 6)) # Задаем размер графика (опционально)
    
    # 2. Рисуем график
    plt.plot(predicted_data[0], predicted_data[1], marker='o', linestyle='-', color='blue') # 'o' - кружки, '-' - линия
    plt.plot(given_data[0], given_data[1], marker='o', linestyle='-', color='green')
    # 3. Настройка графика
    plt.title("Graph of sin")
    plt.xlabel("line X")
    plt.ylabel("line Y = sin(X)")
    plt.grid(True) # Включаем сетку (опционально)
    
    # 4. Показать график
    plt.show()


def main():

    
    for i in range(4):
        nn_data_x = np.array([[i / 100] for i in range(0, 1_01)])  # X data
        nn_data_y = np.array([gauss(x) for x in nn_data_x])
        train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(60)]
        train_idx.sort()
        val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
        val_idx.sort()
        val_data_x = nn_data_x[val_idx, :]  # validation X data
        val_data_y = nn_data_y[val_idx, :]  # validation Y data
        nn_data_x = nn_data_x[train_idx, :]  # X data
        nn_data_y = nn_data_y[train_idx, :]  # Y data
        start_time = time.perf_counter()
        best_loss, best_epoch, best_loss_func, best_opt, best_net = genetic_search(
            input_size = 1,
            output_size = 1,
            train_data = (nn_data_x, nn_data_y),
            val_data = (val_data_x, val_data_y), # now it should be not null always and not empty
            NNepoch = 200, # epohs for train NN
            neuron_num = {i for i in range(1, 10)}, # num of neurons in layer from nn_code
            activ_func = {5, 6}, # activation func ind from nn_code
            min_layers = 2,
            max_layers = 5,
            pop_size = 50,
            ngen = 20,
            logToConsole = True)
        end_time = time.perf_counter()
        genetic_search_time = end_time - start_time
        print(f"gs_{i}: ", genetic_search_time, best_loss)
    
    for i in range(4):
        nn_data_x = np.array([[i / 100] for i in range(0, 1_01)])  # X data
        nn_data_y = np.array([gauss(x) for x in nn_data_x])
        train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(60)]
        train_idx.sort()
        val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
        val_idx.sort()
        val_data_x = nn_data_x[val_idx, :]  # validation X data
        val_data_y = nn_data_y[val_idx, :]  # validation Y data
        nn_data_x = nn_data_x[train_idx, :]  # X data
        nn_data_y = nn_data_y[train_idx, :]  # Y data


        start_time = time.perf_counter()
        best_loss, best_epoch, best_loss_func, best_opt, best_net = random_search(
            input_size = 1,
            output_size = 1,
            data = (nn_data_x, nn_data_y),
            opt = "Adam",
            loss = "MeanSquaredError",
            iterations=157,
            min_epoch=200,
            max_epoch=200,
            val_data = (val_data_x, val_data_y), # now it should be not null always and not empty
            callbacks=None,
            nn_min_length=2,
            nn_max_length=5,
            nn_alphabet=["".join(elem) for elem in product([str(i) for i in range(1, 10)], ["5", "6"])])
        end_time = time.perf_counter()
        
        random_search_time = end_time - start_time
        print(f"rs_{i}: ", random_search_time, best_loss)
 

    #print("gs: ", genetic_search_time)
    #print("rs: ", random_search_time)
    #print("rs loss: ", best_loss)
    #print("fs: ", genetic_search_time)
    #print("sas: ", genetic_search_time)
    #
    #
    #
    #
    #
    #netResulted = IModel(1, [1],  1)
    #
    #netResulted.from_dict(best_net)
    #nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])
    #nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
    #predicted_y_values = netResulted.predict(nn_data_x)
    #plotGraph((nn_data_x, nn_data_y), (nn_data_x, predicted_y_values))
    i = 0
    input(i)



if __name__ == "__main__":
    main()
    