# import the necessary libraries
import numpy as np
# import the necessary functions
from generate_population import generate_initial_population
from image_helpers import read_image, show_results, plot_comparison
from calculate_aptitude import evaluate_population, image_objective_function, image_objective_function_spatial_entropy, calculate_shannon_entropy, calculate_spatial_entropy, apply_sigmoid
from tournament_parent_selection import tournament_selection_maximize
from sbx_crossover import sbx
from polynomial_mutation import apply_polynomial_mutation



def solve_GA_contrast_optimization(generations: int):   

    # 0. Generate initial population.
    population = generate_initial_population(population_size, variables, limits)
     
    for generation in range(generations):        
        print(f'Generation {generation+1}/{generations}')
        
        # 1. Calculate aptitude vector of the population.
        aptitude = evaluate_population(population.copy(), objetive_function, image.copy())
        # 2. Get the best individual of the current population.
        best_individual_population = population[np.argmax(aptitude)].copy()
                
        # 3. Select the parents using tournament selection.
        population = tournament_selection_maximize(population.copy(), aptitude)
        
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
               

    # get the final aptitude vector
    aptitude = evaluate_population(population, objetive_function, image.copy())
    # get the best individual index = max(aptitude)
    best_individual_index = np.argmax(aptitude)
    
    # return the object with the best individual and its aptitude after all generations
    return {'individual': population[best_individual_index], 'aptitude': aptitude[best_individual_index]}



# ===============================================================
# GENERAL CONFIGURATIONS
# ===============================================================
# => Image configurations
img_path = 'assets/kodim23.png'
img_height = 500
# => Objective function  ('shannon_entropy' or 'spatial_entropy')
ob = 'spatial_entropy'  

# => Genetic Algorithm configurations
population_size = 50 #np -> Size of the population
generations = 40 #ng -> Number of generations
variables = 2 #nVar -> Number of variables of each individual
limits = np.array([[0, 1],   # Limits for variable 1
                [0, 10]]) # Limits for variable 2

# ===============================================================
# SBX CONFIGURATIONS
# ===============================================================
sbx_prob = 0.8 #pc -> Probability of crossover
sbx_dispersion_param = 4 #nc -> Distribution index (ideal 1 or 2)    

# ===============================================================
# POLYNOMIAL MUTATION CONFIGURATIONS
# ===============================================================
mutation_probability_param = 0.6 #r -> Probability of mutation
distribution_index_param = 95 #nm -> Distribution index ( ideal 20-100)        

# ===============================================================
# PRE-PROCESSING (read the image and calculate the entropy)
# ===============================================================
# 1. define the objective function to use over the GA (shannon_entropy or spatial_entropy).
objetive_function = image_objective_function if ob == 'shannon_entropy' else image_objective_function_spatial_entropy
# 2. prepare the image.
image = read_image(img_path, img_height)
# 3. calculate the entropy of the original image.
original_entropy = calculate_shannon_entropy(image) if ob == 'shannon_entropy' else calculate_spatial_entropy(image)


# Show the configurations
print(f'{"*"*50}')
print('=> Image configurations:')
print(f'Image path: {img_path}')
print(f'Image shape: {image.shape}')
print(f'Original image entropy: {original_entropy}')
print(f'Objective function: {ob}')
print('=> Algorithm configurations:')
print(f'Number of generations: {generations}')
print(f'Population size: {population_size}')
print(f'Number of variables: {variables}')
print(f'Limits: \n{limits}')
print(f'SBX Probability (pc): {sbx_prob}')
print(f'SBX Dispersion Parameter (nc): {sbx_dispersion_param}')
print(f'Mutation Probability: {mutation_probability_param}')
print(f'Distribution Index (nm): {distribution_index_param}')


# ===============================================================
# EXECUTION OF GENETIC ALGORITHM
# ===============================================================    
result = solve_GA_contrast_optimization(generations)

# ===============================================================
# POST-PROCESSING
# ===============================================================
# apply the sigmoid to the image with the best individual and then calculate the entropy
print(f'\tIndividual: {result["individual"]}, Aptitude = {result["aptitude"]}')
image_improved = apply_sigmoid(image.copy(), result['individual'][0], result['individual'][1])
imp_entropy = calculate_shannon_entropy(image_improved) if ob == 'shannon_entropy' else calculate_spatial_entropy(image_improved)
# show the results in a plot comparison (original image vs best image)
plot_comparison(image, image_improved, original_entropy, imp_entropy, result['individual'][0], result['individual'][1])