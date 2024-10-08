import numpy as np
import cv2

"""
@breaf: This function is used to calculate the aptitude of the individuals in the population using any Objective Function passed as a parameter
@context: General approach to evaluate the individuals in the population
@param population: np.ndarray -> All population to be evaluated -> F.E: [[2.3, 4.5], [1.4, -0.2]]
@param objective_function: function -> Objective function to evaluate the individuals
@param of_params: Any additional parameter needed by the objective function. could change depending on the objective function
@return: np.ndarray -> Aptitude of the individuals in the population (Vector) [9, 5]
"""
def evaluate_population(population: np.ndarray, objective_function, **of_params) -> np.ndarray:
    n_individuals, n_var = population.shape
    aptitude = np.zeros(n_individuals)
    for i in range(n_individuals):
        individual = population[i]
        aptitude[i] = objective_function(individual, **of_params)
    return aptitude


def apply_sigmoid(img, alpha: float, delta: float)-> float:
    output_img = 1 / (1 + np.exp(alpha * (delta - img)))
    output_img = cv2.normalize(output_img, None, 0, 1, cv2.NORM_MINMAX)
    return output_img


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        img[:, :, 0] = clahe.apply(img[:, :, 0])
    else:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
    return img.astype(np.float64) / 255.0


def calculate_shannon_entropy(img) -> float:
    img_flat = img.flatten()
    hist_counts, _ = np.histogram(img_flat, bins=256, range=(0, 1), density=False)
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))    


def obj_func_shannon_entropy(individual: np.ndarray, pdi_function: str, image: np.ndarray) -> float:
    value_1, value_2 = individual
    if pdi_function == 'sigmoid':
        # value_1 = alpha, value_2 = delta
        improved_image = apply_sigmoid(image, value_1, value_2)
    elif pdi_function == 'clahe':
        # value_1 = clip_limit, value_2 = tile_grid_size
        value_2 = int(np.round(value_2)) 
        improved_image = apply_clahe(image, value_1, value_2)
    else:
        raise ValueError(f'The function {pdi_function} is not implemented')
    
    return calculate_shannon_entropy(improved_image)


def calculate_spatial_entropy(img: np.ndarray) -> float:
    grad_mags = np.zeros(img.shape[:2])    
    for i in range(3):
        channel = img[:, :, i]
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mags += grad_mag
        
    grad_mags = grad_mags / np.max(grad_mags)
    hist_counts, _ = np.histogram(grad_mags.flatten(), bins=256, range=(0, 1), density=False)
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def obj_func_spatial_entropy(individual: np.ndarray, pdi_function: str, image: np.ndarray) -> float:
    value_1, value_2 = individual
    if pdi_function == 'sigmoid':
        improved_image = apply_sigmoid(image, value_1, value_2)
    elif pdi_function == 'clahe':
        value_2 = int(np.round(value_2)) 
        improved_image = apply_clahe(image, value_1, value_2)
    else:
        raise ValueError(f'The function {pdi_function} is not implemented')
    return calculate_spatial_entropy(improved_image)



if __name__ == '__main__':
    # JUST FOR TESTING
    # simulate a population
    population = np.array([[2.3 , 4.5], [1.4, 2]])
    # simulate an image reading one from assets
    img = cv2.imread('./assets/kodim23.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.0
    # pass the parameters using kwargs
    params = {
        'pdi_function': 'clahe', # sigmoid or clahe
        'image': img
    }
    # evaluate the population
    aptitude = evaluate_population(population, obj_func_shannon_entropy, **params)
    # get the index of the max aptitude
    best_index = np.argmax(aptitude)    
    
    print(f'Population: \n{population}')
    print(f'Aptitude Vector: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {population[best_index]}')   