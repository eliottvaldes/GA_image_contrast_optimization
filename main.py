from ga_complete import run_ga

# => Image configurations
img_path = 'assets/bones/4370.png'
img_max_height = 900

# define the Digital Image Processing function to be used to transform the image
dip_function = 'clahe' # 'sigmoid' or 'clahe'

# define the limits dynamically
limits = []
if dip_function == 'sigmoid':
    # limits 4 sigmoid function
    limits = [[0, 1], [0, 10]]
else:
    # limits 4 clahe function
    limits = [[1, 4], [1, 30]]
    if 'bones' in img_path:
        # new limits for bones images 2 get better results
        limits = [[4, 11], [1, 50]]
        

# GA configuration
ga_config = {
    'population_size': 150,
    'generations': 10,
    'variables': 2,
    'limits': limits,
    'sbx_prob': 0.8,
    'sbx_dispersion_param': 3,
    'mutation_probability_param': 0.7,
    'distribution_index_param': 95,
    # extra configuration
    'dip_function_4_improvement': dip_function,
    'objetive_function': 'spatial_entropy', # 'spatial_entropy' or 'shannon_entropy'
    'parent_selection_optimization': 'maximize', # 'maximize' or 'minimize'
    'image_path': img_path,
    'img_max_height': img_max_height,
    'dynamic_sbx_increasing': True,
    'early_stop': False,
}

save_log = True
show_image_result = True

# =======================================
# RUN THE GA
# =======================================
run_ga(ga_config, save_log, show_image_result)