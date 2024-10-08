"""
@brief: Save the results in a csv file
@description: This function saves the results in a csv file. If the file already exists, it appends the new row to the file.
@param: ga_result: dict - dictionary with the results of the genetic algorithm
@param: ga_config: dict - dictionary with the configuration of the genetic algorithm
@param: folder_path: str - path to save the results
@return: None
"""
def save_results(ga_result: dict, ga_config: dict,  folder_path: str) -> None:
    import os
    from datetime import datetime
    import pandas as pd

    try:
        
        # FORMAT THE RESULTS
        results = {}
        for i, value in enumerate(ga_result['individual']):
            results[f'individual_variable_{i+1}'] = value
        results['aptitude'] = ga_result['aptitude']
        results.update(ga_config)
        results['limits'] = str(list(results['limits']))
        results['image_height'] = results['image'].shape[0]
        results.pop('image')                    
        if hasattr(results['objetive_function'], '__name__'):
            results['objetive_function'] = results['objetive_function'].__name__
        if hasattr(results['parent_selection_optimization'], '__name__'):
            results['parent_selection_optimization'] = results['parent_selection_optimization'].__name__
    
        # SAVE THE RESULTS
        # Create the folder if it doesn't exist
        folder = os.path.dirname(folder_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # define the file name using the image_path name or the current date and time
        file_name = results.get('image_path').split('/')[-1] if results.get('image_path') else datetime.now().strftime("%Y%m%d_%H%M%S")
        #remove the file extension
        file_name = file_name.split('.')[0]
        
        # validate if the file_name exists in the folder path. if it exists, open and add the new row to the file. if not, create a new file and add the row.
        if os.path.exists(f"{folder_path}{file_name}.csv"):
            # open the file
            df = pd.read_csv(f"{folder_path}{file_name}.csv")
            # add the new row to the existing file
            df = df._append(results, ignore_index=True)
            df.to_csv(f"{folder_path}{file_name}.csv", index=False)
            print(f"Results updated in {folder_path}{file_name}.csv")
        else:
            df = pd.DataFrame(results, index=[0])
            df.to_csv(f"{folder_path}{file_name}.csv", index=False)
            print(f"Results saved in {folder_path}{file_name}.csv")
            
    except Exception as e:
        print(f"Error saving the results: {e}")
        raise e



# function to show the results.
"""
@brief: Plot the results of the genetic algorithm optimization
@description: This function plots the original image and the improved image using the best individual found by the genetic algorithm.
@param: img_original: np.ndarray - original image
@param: ga_result: dict - dictionary with the results of the genetic algorithm
@param: tmp_objetive_function: str - name of the objective function used in the genetic algorithm.
@param: dip_function_selected: str - name of the digital image processing function used in the genetic algorithm.
@return: None
"""
def plot_results(img_original, ga_result: dict, tmp_objetive_function: str, dip_function_selected: str) -> None:
    import matplotlib.pyplot as plt
    from calculate_aptitude import apply_sigmoid, apply_clahe, calculate_shannon_entropy, calculate_spatial_entropy
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    import os

    # Calculate the entropy of the original image
    original_entropy = calculate_spatial_entropy(img_original) if tmp_objetive_function == 'obj_func_spatial_entropy' else calculate_shannon_entropy(img_original)
    
    # improve the image
    variable_1, variable_2 = ga_result['individual']
    img_improved = img_original.copy()
    if dip_function_selected == 'sigmoid':
        img_improved = apply_sigmoid(img_improved, variable_1, variable_2)
        image_description = f"Sigmoid function. \nalfa = {variable_1} and delta = {variable_2}\nEntropy: {tmp_objetive_function}"
    elif dip_function_selected == 'clahe':
        variable_2 = int(np.round(variable_2))
        img_improved = apply_clahe(img_improved, variable_1, variable_2)
        image_description = f"CLAHE function. \nclip limit = {variable_1} and grid size = {variable_2}\nEntropy: {tmp_objetive_function}"
    else:
        raise ValueError(f'The function {dip_function_selected} is not implemented')

    # calculate the entropy of the improved image
    best_image_entropy = calculate_spatial_entropy(img_improved) if tmp_objetive_function == 'obj_func_spatial_entropy' else calculate_shannon_entropy(img_improved)
    
    # Create the figure and the grid of images
    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0, share_all=True)

    # Plot Original image
    grid[0].imshow(img_original, cmap='gray')
    grid[0].set_title(f'Original Image\nEntropy: {original_entropy}')

    # Plot Improved image
    grid[1].imshow(img_improved, cmap='gray')
    grid[1].set_title(f'Best Image\nEntropy: {best_image_entropy}\n{image_description}')

    # Deactivate the axis
    for ax in grid:
        ax.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    
    # configuration for local test
    import numpy as np
    import cv2
        
    file_result_path= './assets/results/'    
    result={'individual': [0.9087878718730255, 6.473717473195805], 'aptitude': 4.97570161859357}
    ga_config={'population_size': 10, 'generations': 1, 'variables': 2, 'limits': np.array([[ 0,  1],[ 0, 10]]), 'sbx_prob': 0.8, 'sbx_dispersion_param': 3, 'mutation_probability_param': 0.7, 'distribution_index_param': 95, 'objetive_function': "spatial_entropy", 'parent_selection_optimization': "maximaze", 'image': np.array([1,2,3,4,5]), 'image_path': 'assets/microphotographs-of-pulmonary-blood-vessels.png'}        
    # save the results
    save_results(result, ga_config, file_result_path)
    
    # plot the results
    img = cv2.imread('./assets/microphotographs-of-pulmonary-blood-vessels.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.0
    plot_results(img, result, 'spatial_entropy', 'sigmoid')
