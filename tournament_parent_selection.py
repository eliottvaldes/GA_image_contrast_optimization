import numpy as np


"""
@brief: Selects parents using tournament selection for crossover.
@args: population (np.ndarray): The current population of individuals.
@args: aptitude (np.ndarray): The aptitude (fitness) of the individuals.
@args: num_parents (int): The number of parents to select for crossover.
@return: np.ndarray: The selected parents for crossover.
"""
def tournament_selection(population: np.ndarray, aptitude: np.ndarray) -> np.ndarray:    
    num_parents, num_variables = population.shape    
        
    print(f'Shape Population: {population.shape}')
    print(f'Num parents: {num_parents}')
    print(f'Num individuals: {num_variables}')
    
    parents = np.zeros((num_parents, num_variables))    

    for i in range(num_parents):
        # Select two random individuals
        idx1 = np.random.randint(num_parents)
        idx2 = np.random.randint(num_parents)
        # Compare the aptitude of the two individuals
        if aptitude[idx1] < aptitude[idx2]:
            parents[i] = population[idx1]
        else:
            parents[i] = population[idx2]
    
    # Return the selected parents
    return parents

if __name__ == '__main__':
    # Example usage
    population_size = 5  # Size of the population
    variables = 2        # Number of variables per individual
    limits = np.array([[1, 3],   # Limits for variable 1
                    [-1, 5]]) # Limits for variable 2

    initial_population = generate_initial_population(population_size, variables, limits)
    print(f'Population_generated: \n{initial_population}\n')
    
    # evaluate the population
    aptitude = evaluate_population(initial_population, langermann_function)
    # get the index of the minimum aptitude - BEST INDIVIDUAL
    best_index = np.argmin(aptitude)
    
    print(f'Aptitude: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {initial_population[best_index]}')
    
    # Seleccionar padres usando el torneo    
    selected_parents = tournament_selection(initial_population, aptitude)
    print(f'\nPadres seleccionados: \n{selected_parents}')
