import numpy as np


"""
@breaf: This function generates an initial population for a Genetic Algorithm. Creates REAL VALUES individuals
@context: General approach to generate an initial population for a GA
@param population_size: int -> Size of the population
@param variables: int -> Number of variables per individual
@param limits: np.ndarray -> Limits of the variables -> F.E: [[1, 3], [-1, 5]]
"""
def generate_initial_population(population_size: int, variables: int, limits: np.ndarray) -> np.ndarray:
    # Create an initial population matrix filled with zeros
    population = np.zeros((population_size, variables))
    
    # Fill the matrix with random values within the specified limits
    for i in range(population_size):
        for j in range(variables):
            lb = limits[j][0]
            up = limits[j][1]
            population[i, j] = lb + np.random.rand() * (up - lb)
    
    return population



if __name__ == '__main__':
    # Example usage
    population_size = 100  # Size of the population
    variables = 2        # Number of variables per individual
    limits = np.array([[0, 1],   # Limits for variable 1
                    [0, 10]]) # Limits for variable 2

    initial_population = generate_initial_population(population_size, variables, limits)
    print(f'Population_generated: \n{initial_population}')
