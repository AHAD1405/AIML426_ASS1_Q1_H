import random
import numpy as np
#from deap import base, creator, tools, algorithms
import pandas
import pandas as pd
import os

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

    # Read file and extract data file
    full_path = os.path.abspath(__file__) # Get the full path of the script     
    script_directory = os.path.dirname(full_path) # Get the directory of the script
    data_file = os.path.join(script_directory,file_name) # Get the full path of the data file

    with open(data_file,'r') as file: 
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
        elif max_capacity == 10000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}

    return item_dict, max_capacity, item_count, optimal_value

def fitness(individual, items, max_weight):
    """
        Function: This function calculates the 
                    fitness score (total value) of an individual solution. If the bit is 1, it adds the corresponding item's weight and 
                    value to the running totals.
        Return:   If the total weight exceeds the maximum weight capacity, it returns 0 (invalid solution). Otherwise, it returns the 
                    total value       
    """
    weight = 0
    value = 0
    for i, item in enumerate(individual):
        if item:
            weight += items['weights'][i]
            value += items['values'][i]
    if weight > max_weight:
        return 0
    return value

def tournament_selection(population, fitness_values, tournament_size):
    """
    Perform tournament selection to produce a number of parents equal to the number of individuals in the population.

    Parameters:
    population (list): The population of individuals.
    fitness_values (list): The fitness values of the individuals.
    tournament_size (int): The number of individuals to participate in each tournament.

    Returns:
    list: A list of selected parents.
    """
    selected_parents = []
    population_copy = population[:]
    fitness_values_copy = fitness_values[:]
    
    while len(selected_parents) < len(population):
        current_tournament_size = min(tournament_size, len(population_copy))
        selected_indices = random.sample(range(len(population_copy)), current_tournament_size)
        tournament_individuals = [population_copy[i] for i in selected_indices]
        tournament_fitness_values = [fitness_values_copy[i] for i in selected_indices]
        
        best_individual_index = tournament_fitness_values.index(max(tournament_fitness_values))
        selected_parents.append(tournament_individuals[best_individual_index])
        
        # Remove the selected individual from the population and fitness values
        del population_copy[selected_indices[best_individual_index]]
        del fitness_values_copy[selected_indices[best_individual_index]]
    
    return selected_parents

def crossover(parents, num_items):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        crossover_point = random.randint(1, num_items-1)  # It is used to select a random crossover point for the crossover operation
        child1 = parent1[:crossover_point] + parent2[crossover_point:]  # It creates a new list by concatenating two sublists of the original list.
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring

def mutation(offspring, num_items, mutation_rate):
    for individual in offspring:
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, num_items -1)
                individual[mutation_point] = 1 - individual[mutation_point]
    return offspring

def print_summary(best_values, best_weights, values_mean, weights_mean, values_std, weights_std, dataset_file, optimal_value):
    """
    Print the summary of the results.

    Args:
        best_values (list): A list containing the best values obtained in each run.
        best_weights (list): A list containing the weights of the best solutions obtained in each run.
        values_mean (float): The mean of the best values.
        weights_mean (float): The mean of the weights of the best solutions.
        values_std (float): The standard deviation of the best values.
        weights_std (float): The standard deviation of the weights of the best solutions.
    """
    # run sequence column
    runs_column = list(['Run '+ str(i) for i in range(1, len(best_values)+1)])

    # Create a dictionary with the two lists as values
    data = {'': runs_column, 'Total value': best_values, 'Total Wight': best_weights}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'Total value': values_mean, 'Total Wight': weights_mean})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Total value': values_std, 'Total Wight': weights_std})
    data_table = pd.concat([data_table, std_row], ignore_index=True)
    print(f'Dataset: {dataset_file}, Optimal Value: {optimal_value}')
    print(data_table)

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

    dataset_files = ['23_10000', '10_269','100_995']  
    
    for dataset_file in dataset_files:
        knapsack_items, max_capacity, num_items, optimal_value = create_dataset(dataset_file)  # Obtain dataset values into parameter
        
        # Reset the best_individual, best_weight, and best_value lists
        best_individual = []
        best_weight = []
        best_value = []

        for run in range(5):
            random.seed(seed_[run])
            
            # Initialize random individual. then add it to population 
            population = []
            for _ in range(population_size):
                individual = [random.randint(0, 1) for _ in range(num_items)]  # It generates a random integer between 0 and 1 (inclusive)
                population.append(individual)

            # Genetic algorithm
            for generation in range(num_generations):
                # Evaluate fitness
                fitness_scores = [fitness(individual, knapsack_items, max_capacity) for individual in population]
                
                # apply elitism: select the top 1 individuals from the population
                #elite_idx = np.argsort(fitness_scores)[::-1][:elitism_size]
                #best_elitism_individuals = [population[i] for i in elite_idx]

                # PARENT SELECTION: select parent using tournament selection
                parent_individuals = []
                parent = tournament_selection(population, fitness_scores, tournament_size)
                [parent_individuals.append(individual) for individual in parent]

                # CROSSOVER
                offspring = crossover(parent_individuals, num_items)
                # MUTATION
                offspring = mutation(offspring, num_items, mutation_rate)
                
                # REPRODUCTION: Create and evaluate population for next generation, It is used to select the fittest individuals for the next generation
                offspring_fitness = [fitness(individual, knapsack_items, max_capacity) for individual in offspring]
                offspring_sorted = [individual for individual, score in sorted(zip(offspring, offspring_fitness), key=lambda x: x[1], reverse=True)]
                #population = best_elitism_individuals + offspring_sorted[:len(population)-elitism_size]
                population = offspring_sorted

                # Stop Crieteria: if hte best solution during this generation equal optimal value, then break the loop and go to next run 
                if max(offspring_fitness) == optimal_value: 
                    break
    
            # Find the best solution and store its (weight, value, and individual) each run
            best_individual.append(max(population, key=lambda individual: fitness(individual, knapsack_items, max_capacity))) # returns the individual with the maximum fitness score from the final population.
            best_value.append(fitness(best_individual[run], knapsack_items, max_capacity))
            best_weight.append(sum([knapsack_items['weights'][i] * best_individual[run][i] for i in range(len(knapsack_items['weights']))]))

        # Calulate Mean and Std for value and weights
        mean_value = round(np.mean(best_value), 2)
        mean_wight = round(np.mean(best_weight))
        std_value = round(np.std(best_value))
        std_wight = round(np.std(best_weight))
        
        # PrintTable 
        print_summary(best_value, best_weight, mean_value, mean_wight, std_value, std_wight, dataset_file, optimal_value)


if __name__ == "__main__":
    main()   
