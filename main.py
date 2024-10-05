from ga_complete import run_ga

# => Image configurations
img_path = 'assets/microphotographs-of-pulmonary-blood-vessels.png'
img_max_height = 900

# GA configuration
ga_config = {
    'population_size': 10,
    'generations': 1,
    'variables': 2,
    'limits': list([[0, 1], [0, 10]]),
    'sbx_prob': 0.8,
    'sbx_dispersion_param': 3,
    'mutation_probability_param': 0.7,
    'distribution_index_param': 95,
    'objetive_function': 'spatial_entropy', # 'spatial_entropy' or 'shannon_entropy'
    'parent_selection_optimization': 'maximize', # 'maximize' or 'minimize'
    'image_path': img_path,
    'img_max_height': img_max_height,
}

save_log = True
show_image_result = False

# =======================================
# RUN THE GA
# =======================================
run_ga(ga_config, save_log, show_image_result)