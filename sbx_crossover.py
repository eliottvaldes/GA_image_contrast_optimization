import numpy as np

# define the SBX function
"""
@breaf: This function executes the Simulated Binary Crossover (SBX) operator
@context: SBX simulates the crossover of two parents generating two children without need variables codification/decodification in binary form.
@param parents: np.ndarray -> Parents to be crossed -> F.E: [2.3, 4.5], [1.4, -0.2]]
@param limits: np.ndarray -> Limits of the variables -> F.E: [[1, 3], [-1, 5]]
@param sbx_prob: float -> Probability of crossover (pc)
@param sbx_dispersion: int -> Distribution index (ideal 1 or 2) (nc)
@return: None -> it changes the parents using memory reference
"""
def sbx(parents: np.ndarray, limits: np.ndarray, sbx_prob: float, sbx_dispersion: int) -> None:
    # get the number of parents
    n_parents, n_var = parents.shape    
    # loop over the parents
    for i in range(0, n_parents, 2):
        # generate a random number to check if the parents will be crossed or not
        r = np.random.rand()
        # check if the parents will be crossed
        if r <= sbx_prob:
            # generate a random number
            u = np.random.rand()
            # loop over the variables
            for j in range(n_var):
                # get the variables of the parents
                var1 = parents[i][j]
                var2 = parents[i+1][j]
                
                # avoid errors(avoid division by zero)
                if var1 == var2:
                    continue                    
                              
                # get the lower and upper bounds of the variable                    
                lb = limits[j][0]
                ub = limits[j][1]
                # calculate beta = 1+2/(p2-p1)*min([(p1-lb(j)),(ub(j)-p2)])
                beta = 1 + (2/(var2-var1)) * ( min( (var1 - lb), (ub - var2) ) )
                # calculate alpha = 2-abs(beta)^-(nc+1).
                alpha = 2 - np.power(abs(beta), -(sbx_dispersion+1))                
                # validate alpha vs u
                if u <= 1/alpha:
                    # calculate beta_c = (u*alpha)^(1/(nc+1))
                    beta_c = np.power(u*alpha, 1/(sbx_dispersion+1))
                else:
                    # calculate beta_c = (1/(2-u*alpha))^(1/(nc+1))
                    beta_c = np.power(1/(2-u*alpha), 1/(sbx_dispersion+1))
                
                # replace the parents with the children calculated
                parents[i][j] = 0.5 * ( (var1+var2) - (beta_c*abs(var2-var1)) )
                parents[i+1][j] = 0.5 * ( (var1+var2) + (beta_c*abs(var2-var1)) )
                        


if __name__ == '__main__':
    
    # JUST FOR TESTING
    #GENERAL CONFIGURATIONS
    population_size = 2 #np -> Size of the population
    variables = 2 #nVar -> Number of variables of each individual
    limits = np.array([[0, 1], #limits var 1 -> [Lower, Upper]
                    [0, 10]]) #limits var 2 -> [Lower, Upper]
    #SBX CONFIGURATIONS
    sbx_prob = 0.7 #pc -> Probability of crossover
    sbx_dispersion_param = 7 #nc -> Distribution index (ideal 1 or 2)

    # define the population. initializa with zeros
    population = np.zeros((population_size, variables))

    # define a high sbx_prob for testing
    sbx_prob = 10
    # define a popilation 
    population = np.array([
        [0.98497561, 8.64246315],
        [0.98497561, 7.64246315],
        [0.99754764, 9.16562837],
        [0.99801123, 9.19327856],
        [0.96851182, 8.52015122],
        [0.99995746, 9.5404104 ],
        [0.9974115 , 9.16564782],
        [0.99803069, 9.19314243],
        [0.99490194, 8.61926468],
        [0.99996816, 9.64940138]
 ])
    
    # *hardcode the param u=0.95 to get the same results as the class example


    # execute the sbx function
    print(f'Initial population: \n{population}')
    print(f'Limits: \n{limits}')
    print(f'SBX Probability: {sbx_prob}')
    print(f'SBX Dispersion Parameter: {sbx_dispersion_param}')
    print(f'{"*"*50}')
    print(f'Executing SBX...')
    print(f'{"*"*50}')
    sbx(population, limits, sbx_prob, sbx_dispersion_param)
    print(f'Final population: \n{population}')