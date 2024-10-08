from ga_complete import run_ga

# => Image configurations
img_path = 'assets/White-blood-cell-(basophil).jpg'
img_max_height = 900

# GA configuration
ga_config = {
    'population_size': 50,
    'generations': 10,
    'variables': 2,
    'limits': list([[0, 1], [0, 10]]),
    'sbx_prob': 0.8,
    'sbx_dispersion_param': 3,
    'mutation_probability_param': 0.7,
    'distribution_index_param': 95,
    # extra configuration
    'pdi_function_4_improvement': 'sigmoid', # 'sigmoid' or 'clahe'
    'objetive_function': 'shannon_entropy', # 'spatial_entropy' or 'shannon_entropy'
    'parent_selection_optimization': 'maximize', # 'maximize' or 'minimize'
    'image_path': img_path,
    'img_max_height': img_max_height,
    'dynamic_sbx_increasing': False,
    'early_stop': False,
}

save_log = True
show_image_result = True

# =======================================
# RUN THE GA
# =======================================
run_ga(ga_config, save_log, show_image_result)