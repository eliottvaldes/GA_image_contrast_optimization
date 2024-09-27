import numpy as np
import cv2

"""
@breaf: This function is used to calculate the aptitude of the individuals in the population using any Objective Function passed as a parameter
@context: General approach to evaluate the individuals in the population
@param population: np.ndarray -> All population to be evaluated -> F.E: [[2.3, 4.5], [1.4, -0.2]]
@param objective_function: function -> Objective function to evaluate the individuals
@return: np.ndarray -> Aptitude of the individuals in the population (Vector) [9, 5]
"""
def evaluate_population(population: np.ndarray, objective_function, of_params) -> np.ndarray:
    # get the number of individuals and variables
    n_individuals, n_var = population.shape
    # create the aptitude vector
    aptitude = np.zeros(n_individuals)
    # loop over the population
    for i in range(n_individuals):
        # get the individual
        individual = population[i]
        # evaluate the individual
        aptitude[i] = objective_function(individual, of_params)
    return aptitude


def apply_sigmoid(img, alpha: float, delta: float)-> float:
    output_img = 1 / (1 + np.exp(alpha * (delta - img)))
    # Reescalar la imagen al rango [0, 1]
    output_img = cv2.normalize(output_img, None, 0, 1, cv2.NORM_MINMAX)
    return output_img


def calculate_shannon_entropy(img) -> float:
    # Convertir la imagen a un arreglo 1D
    img_flat = img.flatten()
    # Calcular el histograma (counts)
    hist_counts, _ = np.histogram(img_flat, bins=256, range=(0, 1), density=False)
    # Normalizar para obtener probabilidades
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    # Eliminar ceros para evitar log(0)
    probabilities = probabilities[probabilities > 0]
    # Calcular la entropía de Shannon
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    return shannon_entropy    


def image_objective_function(individual: np.ndarray, image: np.ndarray) -> float:
    # get the alpha and delta values
    alpha, delta = individual
    # apply the sigmoid to the image
    sigmoid_image = apply_sigmoid(image, alpha, delta)
    # calculate the shannon entropy
    return calculate_shannon_entropy(sigmoid_image)    
    



if __name__ == '__main__':
    # JUST FOR TESTING
    # GENERAL CONFIGURATIONS
    population = np.array([[2.3 , 4.5], [1.4, -0.2]])
    
    # evaluate the population
    aptitude = evaluate_population(population, image_objective_function, [np.random.rand(400, 400)])
    # get the index of the minimum aptitude
    best_index = np.argmin(aptitude)
    
    print(f'Population: \n{population}\n')
    print(f'Aptitude: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {population[best_index]}')   