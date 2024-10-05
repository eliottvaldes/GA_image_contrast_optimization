import os
from datetime import datetime
import pandas as pd


"""
@brief: Save the results in a csv file
@description: This function saves the results in a csv file. If the file already exists, it appends the new row to the file.
@param: results (dict): The object with the results to save.
@param: folder_path (str): The path where the file will be saved.
@return: None
"""
def save_results(results: dict, folder_path: str) -> None:
    try:
        # Create the folder if it doesn't exist
        folder = os.path.dirname(folder_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # define the file name using the image name or the current date and time
        file_name = results.get('image').split('/')[-1] if results.get('image') else datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # create the object with the results 
    ga_log= {'individual_variable_1': 0.9565342695753656, 'individual_variable_2': 9.965263017153724, 'aptitude': 4.982804479327174, 'population_size': 10, 'generations': 1, 'variables': 2, 'limits': '[array([0, 1]), array([ 0, 10])]', 'sbx_prob': 0.8, 'sbx_dispersion_param': 3, 'mutation_probability_param': 0.7, 'distribution_index_param': 95, 'objetive_function': 'obj_func_shannon_spatial_entropy', 'parent_selection_optimization': 'tournament_selection_maximize', 'image': 'assets/microphotographs-of-pulmonary-blood-vessels.png', 'image_height': 384}
    file_result_path= './assets/results/'
    
    # save the results
    save_results(ga_log, file_result_path)
