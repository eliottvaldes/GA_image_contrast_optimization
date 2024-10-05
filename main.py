# import the necessary libraries
import numpy as np
# import the necessary functions
from generate_population import generate_initial_population
from image_helpers import read_image
from calculate_aptitude import evaluate_population, obj_func_shannon_entropy, obj_func_shannon_spatial_entropy
from tournament_parent_selection import tournament_selection_minimize, tournament_selection_maximize
from sbx_crossover import sbx
from polynomial_mutation import apply_polynomial_mutation


"""
@breif: This runs the Genetic Algorithm to solve any optimization problem for real-valued functions. Particularly, it solves the contrast optimization problem.
@param: ga_config (dict): The configuration object for the Genetic Algorithm.
@return: dict: The object with the best individual and its aptitude after all generations.
"""
def solve_GA_contrast_optimization(ga_config: dict):  
    # --------------------------------------------------------------- 
    # Get all configurations
    # ---------------------------------------------------------------
    population_size = ga_config.get('population_size', 50)
    generations = ga_config.get('generations', 40)
    variables = ga_config.get('variables', 2)
    limits = ga_config.get('limits', np.array([[0, 1], [0, 10]]))
    sbx_prob = ga_config.get('sbx_prob', 0.8)
    sbx_dispersion_param = ga_config.get('sbx_dispersion_param', 3)
    mutation_probability_param = ga_config.get('mutation_probability_param', 0.7)
    distribution_index_param = ga_config.get('distribution_index_param', 95)
    objetive_function = ga_config.get('objetive_function', obj_func_shannon_entropy)    
    parent_selection_optimization = ga_config.get('parent_selection_optimization', tournament_selection_maximize)
    image = ga_config.get('image', None)
    
    # Dynamic configurations
    # Define the increment_rate for the sbx_dispersion_param. It's calculated based on the number of generations+(population_size/variables**variables). Max value of sbx_dispersion_param is 20.
    increment_rate = (20 - sbx_dispersion_param) / (generations+(population_size/variables**variables)) if sbx_dispersion_param < 20 else 0
        
    # ---------------------------------------------------------------
    # Start the Genetic Algorithm
    # ---------------------------------------------------------------
    # 0. Generate initial population.
    population = generate_initial_population(population_size, variables, limits)
     
    for generation in range(generations):        
        print(f'Generation {generation+1}/{generations}')
        
        # 1. Calculate aptitude vector of the population.
        aptitude = evaluate_population(population.copy(), objetive_function, image.copy())
        # 2. Get the best individual of the current population.
        best_individual_population = population[np.argmax(aptitude)].copy()
                
        # 3. Select the parents using tournament selection.
        population = parent_selection_optimization(population.copy(), aptitude)
        
        # 4. SBX crossover.
        sbx(population, limits, sbx_prob, sbx_dispersion_param)
        
        # 5. Apply Polynomial Mutation.
        apply_polynomial_mutation(population, limits, mutation_probability_param, distribution_index_param)
        
        # 6. Calculate aptitude after mutation and crossover.
        aptitude_af = evaluate_population(population, objetive_function, image.copy())
        # 6.1 Get the worst individual index of the population (child population).
        worst_aptitude_index = np.argmin(aptitude_af)
                
        # 7. ELITISM - replace the worst individual in the child population with the best individual from the previous generation.
        population[worst_aptitude_index] = best_individual_population.copy()
        
        # EXTRA: increment the sbx_dispersion_param if it is less than 20.
        sbx_dispersion_param += increment_rate if sbx_dispersion_param < 20 else 0
               

    # get the final aptitude vector
    aptitude = evaluate_population(population, objetive_function, image.copy())
    # get the best individual index = max(aptitude)
    best_individual_index = np.argmax(aptitude)
    
    # return the object with the best individual and its aptitude after all generations
    return {'individual': population[best_individual_index], 'aptitude': aptitude[best_individual_index]}



if __name__ == '__main__':
    # ---------------------------------------------------------------
    # CONFIGURATIONS
    # ---------------------------------------------------------------
    # => Image configurations
    img_path = 'assets/microphotographs-of-pulmonary-blood-vessels.png'
    img_max_height = 900
    # => Objective function to use ('shannon_entropy' or 'spatial_entropy')
    ob_config = 'spatial_entropy'
    # => Select the optimization goal ('maximize' or 'minimize')
    parent_selection_optimization = 'maximize'

    image = read_image(img_path, img_max_height)

    # Create the configuration object
    ga_config = {
        'population_size': 50,
        'generations': 40,
        'variables': 2,
        'limits': np.array([[0, 1], [0, 10]]),
        'sbx_prob': 0.8,
        'sbx_dispersion_param': 3,
        'mutation_probability_param': 0.7,
        'distribution_index_param': 95,
        'objetive_function': obj_func_shannon_entropy if ob_config == 'shannon_entropy' else obj_func_shannon_spatial_entropy,
        'parent_selection_optimization': tournament_selection_maximize if parent_selection_optimization == 'maximize' else tournament_selection_minimize,
        'image': image
    }


    # ---------------------------------------------------------------
    # EXECUTE THE GENETIC ALGORITHM
    # ---------------------------------------------------------------
    print(f"Running the Genetic Algorithm to solve the contrast optimization problem...")
    result = solve_GA_contrast_optimization(ga_config)
    print(f"{result= } ")