import random
import numpy as np
#from deap import base, creator, tools, algorithms
import pandas

def create_dataset(file_name):
    """
        Reads a dataset from a file and extracts weights, values, maximum capacity, item count, and optimal value.

        Parameters:
        file_name (str): The name of the file containing the dataset.

        Returns:
        tuple: A tuple containing:
            - item_dict (dict): A dictionary with two keys:
                - "weights" (list): A list of weights of the items.
                - "values" (list): A list of values of the items.
            - max_capacity (int): The maximum capacity of the knapsack.
            - item_count (int): The number of items.
            - optimal_value (int): The optimal value for the given maximum capacity.
    """
    weights = []
    values = []
    max_capacity = 0   # value of maximume weights
    item_count = 0     # number of the items for each individual
    optimal_value = 0  

    with open(file_name,'r') as file: 
        data = file.readlines()      

        for idx, line in enumerate(data): # extract weights and vlues and store it into list
            x = line.split()
            if idx == 0:
                max_capacity = int(x[1])
                item_count = int(x[0])
            else:
                weights.append(int(x[1]))
                values.append(int(x[0]))
        
        # Find the vlaue of optimal_value paramener. depend on value of (max_capacity) 
        if max_capacity == 269: optimal_value = 295
        elif max_capacity == 1000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}

    return item_dict, max_capacity, item_count, optimal_value

def main():
    population_size = 50   # Population size 
    num_generations  = 50   # number of generations to run the Genetic Algorithm
    mutation_rate = 0.1
    crossover_rate = 0.9
    tournament_size = 3
    seed_ = [20, 30, 40, 50, 60]
    best_individual = []
    best_weight = []
    best_value = []
    elitism_size = 1

    dataset_file = '10_269'
    
    knapsack_items, max_capacity, num_items, optimal_value = create_dataset(dataset_file)  # Obtain dataset values into parameter
    
    for run in range(5):
        random.seed(seed_[run])
        
        # Initialize random individual. then add it to population 
        population = []
        for _ in range(population_size):
            individual = [random.randint(0, 1) for _ in range(num_items)]  # It generates a random integer between 0 and 1 (inclusive)
            population.append(individual)

if __name__ == "__main__":
    main()   
