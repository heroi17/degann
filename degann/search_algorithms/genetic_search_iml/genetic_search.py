from datetime import datetime
from itertools import product
from typing import List, Tuple

from ..nn_code import alph_n_full, alphabet_activations, decode
from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from ..utils import update_random_generator, log_to_file




from deap import base, creator, tools, algorithms
import numpy as np
import random
from copy import copy
import traceback


from degann.search_algorithms.grid_search import grid_search_step #for generating model and validate it

def genetic_topology_repr_encode(genome: List[Tuple[int, int]]) -> str:
    """return str repr of genome"""
    return "".join(f"{layer:x}{activation:x}" for layer, activation in genome)



def genetic_topology_repr_decode(str_genome_repr: str) -> List[Tuple[int, int]]:
    genome = []
    for i in range(0, len(str_genome_repr), 2):
        layer = int(str_genome_repr[i], 16)
        activation = int(str_genome_repr[i+1], 16)
        genome.append((layer, activation))
    return genome


def evaluate(individual, 
             bestNN: List, 
             topologyHistory: dict[str, float], 
             input_size: int,
             output_size: int,
             train_data: Tuple, 
             val_data: Tuple,
             NNepoch: int
             ): 
    topology_str = genetic_topology_repr_encode(individual)
    
    if topology_str in topologyHistory:
      return (topologyHistory[topology_str],)
    else:
        #(best_loss, best_val_loss, best_net)
        bl, bvl, bn = grid_search_step(
            input_size=input_size, 
            output_size=output_size, 
            code=topology_str, 
            num_epoch=NNepoch,
            opt="Adam",
            loss="MeanSquaredError", 
            data=train_data, 
            val_data=val_data)
        
        topologyHistory[topology_str] = bvl

        bestNN_val = bestNN
        if bestNN_val[1] > bvl:
          bestNN_val[:] = [bn, bvl] # if we don't use [:] then it dosen't chenge the passed object bestNN_val
        return (bl,)




def generate_genome(min_layers, max_layers, neuron_num, activ_func): # min_layers изменен на 2
    num_layers = random.randint(min_layers, max_layers)
    genome = []
    for _ in range(num_layers):
        genome.append((random.choice(list(neuron_num)), random.choice(list(activ_func))))
    return genome


def mutate_individual(individual, indpb, neuron_num, activ_func):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = (random.choice(list(neuron_num)), random.choice(list(activ_func)))
    return individual,

def mate(ind1, ind2, topologyHistory):
    for _ in range(10):
        ind1_pr = copy(ind1)
        ind2_pr = copy(ind2)
        ind1_pr, ind2_pr = tools.cxOnePoint(ind1_pr, ind2_pr)
        if (genetic_topology_repr_encode(ind1_pr) not in topologyHistory and
            genetic_topology_repr_encode(ind1_pr) not in topologyHistory):
            break
        
    ind1 = ind1_pr
    ind2 = ind2_pr
    return ind1, ind2
    

def genetic_search(
    

    #NN train haracteristics
    input_size: int = 1,
    output_size: int = 1,
    train_data: tuple = None,
    val_data: tuple = None, # now it should be not null always and not empty
    NNepoch: int = 10, # epohs for train NN

    #topology haracteristics
    neuron_num = {1, 2, 3, 4}, # num of neurons in layer from nn_code
    activ_func = {1, 2, 3}, # activation func ind from nn_code
    min_layers = 2,
    max_layers = 4,
    pop_size = 10, # size of start population
    ngen = 10, # how epochs for GA 
    


    #special parametrs like logging or callbacks
    logToConsole = False,
    
    ):


    bestNN: List = [{}, float('inf')] # contain here the best NN evere builded here and it's 
    # it is list becouse it should be chengeble when we call
    topologyHistory: dict[str, float] = {} # contain every topology



    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()

    toolbox.register("generate_genome_with_defoult",  # set the func which generate to us the genome
                     generate_genome, 
                     min_layers = min_layers, 
                     max_layers = max_layers, 
                     neuron_num = neuron_num, 
                     activ_func = activ_func)
    
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate_genome_with_defoult)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, bestNN=bestNN, topologyHistory=topologyHistory, input_size=input_size, output_size=input_size, train_data=train_data, val_data=val_data, NNepoch=NNepoch)
    toolbox.register("mate", mate, topologyHistory=topologyHistory) #maybe it's bad war to our topology finder
    toolbox.register("mutate", mutate_individual, indpb=0.01, neuron_num = neuron_num, activ_func = activ_func) #indpb=0.01
    toolbox.register("select", tools.selTournament, tournsize=3)
    

    #for understand(see the evolution)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    #evolution we are need for
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=logToConsole)
    
        best_individual = hof[0]
        print("str repr of the best", genetic_topology_repr_encode(best_individual))
        print(":", best_individual.fitness.values[0])
        print("", sum([layer[0] for layer in best_individual])) 
        
        #now we do not find "opt" and "loss" just use these ones
        return bestNN[1], -1, "MeanSquaredError", "Adam",  bestNN[0]
        
    except Exception as e: #it's for debug of deap
        print(f"Error in evaluate: {e}")
        traceback.print_exc()
        raise Exception

