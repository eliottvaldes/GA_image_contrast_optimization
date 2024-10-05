import os
from datetime import datetime
import pandas as pd


"""
@brief: Save the results in a csv file
@description: This function saves the results in a csv file. If the file already exists, it appends the new row to the file.
@param: ga_result: dict - dictionary with the results of the genetic algorithm
@param: ga_config: dict - dictionary with the configuration of the genetic algorithm
@param: folder_path: str - path to save the results
@return: None
"""
def save_results(ga_result: dict, ga_config: dict,  folder_path: str) -> None:
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
    

if __name__ == "__main__":
    
    # configuration for local test
    import numpy as np
        
    file_result_path= './assets/results/'    
    result={'individual': [0.9087878718730255, 6.473717473195805], 'aptitude': 4.97570161859357}
    ga_config={'population_size': 10, 'generations': 1, 'variables': 2, 'limits': np.array([[ 0,  1],[ 0, 10]]), 'sbx_prob': 0.8, 'sbx_dispersion_param': 3, 'mutation_probability_param': 0.7, 'distribution_index_param': 95, 'objetive_function': 'spatial_entropy', 'parent_selection_optimization': 'maximize', 'image': np.array([1,2,3,4,5]), 'image_path': 'assets/microphotographs-of-pulmonary-blood-vessels.png'}
    
    # save the results
    save_results(result, ga_config, file_result_path)
