# import the necessary libraries
import numpy as np
# import the necessary functions
from generate_population import generate_initial_population
from image_helpers import read_image
from calculate_aptitude import evaluate_population, obj_func_shannon_entropy, obj_func_spatial_entropy
from tournament_parent_selection import tournament_selection_minimize, tournament_selection_maximize
from sbx_crossover import sbx
from polynomial_mutation import apply_polynomial_mutation
# import the function to save the results and plot the images
from results_helpers import save_results, plot_results


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
    objetive_function = ga_config['objetive_function']
    parent_selection_optimization = ga_config['parent_selection_optimization']
    # get the image
    image = ga_config.get('image', None)
    # define the parameters for the objetive function depending on the dip improvement function selected
    dip_function_4_improvement = ga_config.get('dip_function_4_improvement', 'sigmoid')
    of_params = {'pdi_function': dip_function_4_improvement, 'image': image.copy()}
    
    # Dynamic configurations
    activate_dynamic_sbx_increasing = ga_config.get('dynamic_sbx_increasing', False)
    if activate_dynamic_sbx_increasing:
        # Define the increment_rate for the sbx_dispersion_param. It's calculated based on the number of generations+(population_size/variables**variables). Max value of sbx_dispersion_param is 20.
        increment_rate = (20 - sbx_dispersion_param) / (generations+(population_size/variables**variables)) if sbx_dispersion_param < 20 else 0
    else:
        increment_rate = 0
            
    # ---------------------------------------------------------------
    # Start the Genetic Algorithm
    # ---------------------------------------------------------------
    # 0. Generate initial population.
    population = generate_initial_population(population_size, variables, limits)
     
    for generation in range(generations):        
        print(f'Generation {generation+1}/{generations}')
        
        # 1. Calculate aptitude vector of the population.
        aptitude = evaluate_population(population.copy(), objetive_function, **of_params)
        # 2. Get the best individual of the current population.
        best_individual_population = population[np.argmax(aptitude)].copy()
                
        # 3. Select the parents using tournament selection.
        population = parent_selection_optimization(population.copy(), aptitude)
        
        # 4. SBX crossover.
        sbx(population, limits, sbx_prob, sbx_dispersion_param)
        
        # 5. Apply Polynomial Mutation.
        apply_polynomial_mutation(population, limits, mutation_probability_param, distribution_index_param)
        
        # 6. Calculate aptitude after mutation and crossover.
        aptitude_af = evaluate_population(population, objetive_function, **of_params)
        # 6.1 Get the worst individual index of the population (child population).
        worst_aptitude_index = np.argmin(aptitude_af)
                
        # 7. ELITISM - replace the worst individual in the child population with the best individual from the previous generation.
        population[worst_aptitude_index] = best_individual_population.copy()
        
        # EXTRA: increment the sbx_dispersion_param if it is less than 20.
        sbx_dispersion_param += increment_rate if sbx_dispersion_param < 20 else 0
               

    # get the final aptitude vector
    aptitude = evaluate_population(population, objetive_function, **of_params)
    # get the best individual index = max(aptitude)
    best_individual_index = np.argmax(aptitude)
    
    # return the object with the best individual and its aptitude after all generations
    return {'individual': list(population[best_individual_index]), 'aptitude': aptitude[best_individual_index]}



def run_ga(ga_config: dict, save_log: bool, show_image_result: bool = False):    
    # ---------------------------------------------------------------
    # READ THE IMAGE
    # ---------------------------------------------------------------
    img_path = ga_config.get('image_path', 'assets/test.png')
    #get the img_max_height
    img_max_height = ga_config.get('img_max_height', 900)
    # read the image
    image = read_image(img_path, img_max_height)
    
    # ---------------------------------------------------------------
    # CONFIGURE THE GA
    # ---------------------------------------------------------------
    # add the image to the configuration
    ga_config['image'] = image
    # transform the 'limits' lsit to a numpy array
    ga_config['limits'] = np.array(ga_config['limits'])        
    # define the objetice functions and the parent selection optimization    
    ga_config['objetive_function'] = obj_func_spatial_entropy if ga_config.get('objetive_function', 'spatial_entropy') == 'spatial_entropy' else obj_func_shannon_entropy
    ga_config['parent_selection_optimization'] = tournament_selection_maximize if ga_config.get('parent_selection_optimization', 'maximize') == 'maximize' else tournament_selection_minimize
    
    # ---------------------------------------------------------------
    # EXECUTE THE GENETIC ALGORITHM
    # ---------------------------------------------------------------
    print(f"Running the Genetic Algorithm to solve the contrast optimization problem...")
    print(f'{"-"*60}')
    ga_result = solve_GA_contrast_optimization(ga_config)
    print(f'{"-"*60}')
    print(f"End of the Genetic Algorithm. {ga_result= } \n")
    
    # ---------------------------------------------------------------
    # SAVE THE RESULTS AND CONFIGURATION
    # ---------------------------------------------------------------
    if save_log:
        file_result_path = './assets/results/'    
        # save the results    
        save_results(ga_result.copy(), ga_config.copy(), file_result_path)
        
    if show_image_result:
        # show the image
        plot_results(image, ga_result, ga_config['objetive_function'].__name__, ga_config['dip_function_4_improvement'])
        
        
        
if __name__ == '__main__':
    
    import json
           
    # => Image configurations
    img_path = 'assets/White-blood-cell-(basophil).jpg'
    img_max_height = 900

    # define the Digital Image Processing function to be used to transform the image
    dip_function = 'sigmoid' # 'sigmoid' or 'clahe'
    
    # GA configuration  
    ga_config = {
        'population_size': 60,
        'generations': 200,
        'variables': 2,
        'limits': [],
        'sbx_prob': 0.8,
        'sbx_dispersion_param': 3,                              
        'mutation_probability_param': 0.7,
        'distribution_index_param': 95,
        # extra configuration
        'dip_function_4_improvement': dip_function,
        'objetive_function': 'shannon_entropy', # 'spatial_entropy' or 'shannon_entropy'
        'parent_selection_optimization': 'maximize', # 'maximize' or 'minimize'
        'image_path': img_path,
        'img_max_height': img_max_height,
        'dynamic_sbx_increasing': True,
        'early_stop': False,
    }

    save_log = True
    show_image_result = True

    # define the limits dynamically
    if dip_function == 'sigmoid':
        # limits 4 sigmoid function
        limits = [[0, 1], [0, 10]]
    else:
        # limits 4 clahe function
        limits = [[1, 4], [1, 30]]
        if 'bones' in img_path:
            # new limits for bones images 2 get better results
            limits = [[4, 11], [1, 50]]
    ga_config['limits'] = limits

    # ---------------------------------------
    # Show the configuration
    # ---------------------------------------
    print(f"=> GA Configuration: \n{json.dumps(ga_config, indent=4)}")

    # =======================================
    # RUN THE GA
    # =======================================
    run_ga(ga_config, save_log, show_image_result)