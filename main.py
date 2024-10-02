# import the necessary libraries
import numpy as np
# import the necessary functions
from generate_population import generate_initial_population
from image_helpers import read_image, show_results
from calculate_aptitude import evaluate_population, image_objective_function, calculate_shannon_entropy
from tournament_parent_selection import tournament_selection
from sbx_crossover import sbx
from polynomial_mutation import apply_polinomial_mutation



def solve_GA_contrast_optimization(generations: int):   

    population = generate_initial_population(population_size, variables, limits)   
    #print(f'1. Generate initial population.')        
     
    # iterar las generaciones
    for generation in range(generations):
        # 3. get the aptitude vector
        aptitude = evaluate_population(population, image_objective_function, image.copy())
        #print(f'2. Calculate aptitudes.')        
        
        # save the best individual in a variable use a copy
        best_individual = population[np.argmax(aptitude)].copy()
        #print(f'3. Best individual (values): {best_individual}')
        
        # select the parents using tournament selection
        population = tournament_selection(population.copy(), aptitude)
        
        # apply sbx crossover
        sbx(population, limits, sbx_prob, sbx_dispersion_param)
        #print(f'4. SBX crossover.')
        
        # apply mutation
        apply_polinomial_mutation(population, limits, mutation_probability_param, distribution_index_param)
        #print(f'5. Mutation.')
        
        # calculate the new aptitude vector for the population after crossover
        aptitude = evaluate_population(population, image_objective_function, image.copy())
        #print(f'6. Calculate aptitude after mutation')        
        
        # get the worst individual index = min(aptitude)
        worst_after_crossover = np.argmin(aptitude)
        #print(f'7. Worst individual (index): {worst_after_crossover}')
        
        # ELITISM SUSTITUTION
        # replace the worst individual after crossover (children) with the best individual before crossover (parent)
        population[worst_after_crossover] = best_individual
        #print(f'8. Apply elitism: \n\n')


    # get the final aptitude vector
    aptitude = evaluate_population(population, image_objective_function, image.copy())
    # get the best individual index = min(aptitude)
    best_individual_index = np.argmax(aptitude)    
    
    return {'individual': population[best_individual_index], 'aptitude': aptitude[best_individual_index]}



# ===============================================================
#GENERAL CONFIGURATIONS
# ===============================================================
# => Image configurations
img_path = 'assets/4360.png'
img_height = 500
# => Genetic Algorithm configurations
population_size = 30 #np -> Size of the population
generations = 200 #ng -> Number of generations
variables = 2 #nVar -> Number of variables of each individual
limits = np.array([[0, 1],   # Limits for variable 1
                [0, 10]]) # Limits for variable 2

# ===============================================================
#SBX CONFIGURATIONS
# ===============================================================
sbx_prob = 0.9 #pc -> Probability of crossover
sbx_dispersion_param = 2 #nc -> Distribution index (ideal 1 or 2)    

# ===============================================================
# POLINOMIAL MUTATION CONFIGURATIONS
# ===============================================================
mutation_probability_param = 0.7 #r -> Probability of mutation
distribution_index_param = 50 #nm -> Distribution index ( ideal 20-100)        

# prepare the image
image = read_image(img_path, img_height)
# calculate the shannon entropy of the original image
original_entropy = calculate_shannon_entropy(image)

# Show the configurations
print(f'{"*"*50}')
print('=> Image configurations:')
print(f'Image path: {img_path}')
print(f'Image max height: {img_height}px')
print(f'Original image entropy: {original_entropy}')
print('=> Algorithm configurations:')
print(f'Number of generations: {generations}')
print(f'Population size: {population_size}')
print(f'Number of variables: {variables}')
print(f'Limits: \n{limits}')
print(f'SBX Probability (pc): {sbx_prob}')
print(f'SBX Dispersion Parameter (nc): {sbx_dispersion_param}')
print(f'Mutation Probability: {mutation_probability_param}')
print(f'Distribution Index (nm): {distribution_index_param}')
    
result = solve_GA_contrast_optimization(generations)
print(f'\tIndividual: {result["individual"]}, Aptitude = {result["aptitude"]}')

"""
# ===============================================================
# EXECUTION OF THE MAIN FUNCTION
# ===============================================================
print(f'\n{"*"*50}')
print('Running Algorithm...')
results = [] # list of dictionaries to store the results
for i in range(10):
    # show the current execution, the results of function and then add the results to the list
    print(f'Execution {i+1}')
    result = solve_GA_contrast_optimization(generations)
    print(f'\tIndividual: {result["individual"]}, Aptitude = {result["aptitude"]}')
    results.append(result)
else:
    print('Algorithm finished successfully!')    
    
# only pass the results.aptitude of the results dictionaries
partial_results = np.array([result['aptitude'] for result in results])
# get the best, median, worst and standard deviation of the results
best = np.min(partial_results)
median = np.median(partial_results)
worst = np.max(partial_results)
std = np.std(partial_results)

print(f'\n{"*"*50}')
print(f'Statistics:')
print(f'Original image entropy: {original_entropy}')
print(f'Best: {best}')
print(f'Median: {median}')
print(f'Worst: {worst}')
print(f'Standard deviation: {std}')
"""