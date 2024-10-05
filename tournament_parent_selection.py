import numpy as np
import secrets

"""
@brief: Selects parents using tournament selection for crossover.
@args: population (np.ndarray): The current population of individuals.
@args: aptitude (np.ndarray): The aptitude (fitness) of the individuals.
@args: num_parents (int): The number of parents to select for crossover.
@return: np.ndarray: The selected parents for crossover.
"""
def tournament_selection_maximize(population: np.ndarray, aptitude: np.ndarray) -> np.ndarray:    
    num_parents, num_variables = population.shape            
    parents = np.zeros((num_parents, num_variables))    
    for i in range(num_parents):
        idx1 = secrets.randbelow(num_parents)
        idx2 = secrets.randbelow(num_parents)
        # PRIORITIZE THE INDIVIDUAL WITH THE *HIGHEST* APTITUDE
        if aptitude[idx1] > aptitude[idx2]:
            parents[i] = population[idx1]
        else:
            parents[i] = population[idx2]

    return parents


def tournament_selection_minimize(population: np.ndarray, aptitude: np.ndarray) -> np.ndarray:    
    num_parents, num_variables = population.shape            
    parents = np.zeros((num_parents, num_variables))    
    for i in range(num_parents):
        idx1 = secrets.randbelow(num_parents)
        idx2 = secrets.randbelow(num_parents)
        # PRIORITIZE THE INDIVIDUAL WITH THE *LOWEST* APTITUDE
        if aptitude[idx1] < aptitude[idx2]:
            parents[i] = population[idx1]
        else:
            parents[i] = population[idx2]
    
    return parents



if __name__ == '__main__':
    # simulate a population
    population = np.array([
        [0.06545813, 2.47276573],
        [0.46758506, 4.11341923],
        [0.68228584, 3.21668505],
        [0.58070302, 0.94447639],
        [0.17867484, 1.70116833],
        [0.23853537, 2.06979372],
        [0.10239358, 0.91061268],
        [0.76482242, 5.87768613],
        [0.29528902, 4.06142647],
        [0.15634745, 6.05773729],
    ])
    # simulate an aptitude vector
    aptitude = np.array([
        7.60878204, 7.51376325, 7.46352892, 7.59441195, 7.60498951,
        7.60841625, 7.60841625, 7.38423158, 7.56316415, 7.58626385, 
    ])

    print(f'Population: \n{population}')
    print(f'\n{"-"*20}')
    print(f'Example usage of tournament selection for maximization:')
    parents = tournament_selection_maximize(population, aptitude)
    print(f'Selected Parents: \n{parents}')    
    print(f'\n{"-"*20}')
    print(f'Example usage of tournament selection for minimization:')
    parents = tournament_selection_minimize(population, aptitude)
    print(f'Selected Parents: \n{parents}\n')
