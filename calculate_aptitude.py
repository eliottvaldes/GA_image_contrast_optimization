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


def obj_func_shannon_entropy(individual: np.ndarray, image: np.ndarray) -> float:
    alpha, delta = individual
    sigmoid_image = apply_sigmoid(image, alpha, delta)
    return calculate_shannon_entropy(sigmoid_image)


def calculate_spatial_entropy(img: np.ndarray) -> float:
    grad_mags = np.zeros(img.shape[:2])
    for i in range(3):
        channel = img[:, :, i]
        # calculate the gradients in x and y directions
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        # calculate the gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mags += grad_mag  # sum the gradient magnitudes

    # normalize the gradient magnitudes to the range [0, 1]
    grad_mags = grad_mags / np.max(grad_mags)
    # calculate the histogram of gradient magnitudes
    hist_counts, _ = np.histogram(grad_mags.flatten(), bins=256, range=(0, 1), density=False)
    # calculate the probabilities
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    # remove zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    # calculate the spatial entropy
    spatial_entropy = -np.sum(probabilities * np.log2(probabilities))
    return spatial_entropy


def obj_func_shannon_spatial_entropy(individual: np.ndarray, image: np.ndarray) -> float:
    alpha, delta = individual
    sigmoid_image = apply_sigmoid(image, alpha, delta)
    return calculate_spatial_entropy(sigmoid_image)



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