import numpy as np

"""
@breaf: This function is used to mutate each variable of the individual with a polynomial mutation. Applies the mutation to the whole population
@context: Used in continuous GA. Add variability to the individuals in the population
@param population: np.ndarray -> All population to be mutated -> F.E: [2.3, 4.5]
@param limits: np.ndarray -> Limits of the variables -> F.E: [[1, 3], [-1, 5]]
@param mutation_probability: (Pm) int -> Probability of mutation (between 0 and 1)
@param distribution_index: (nm) int -> Distribution index ( ideal 20-100)
@return: None -> it changes the population using memory reference
"""
def apply_polinomial_mutation(population: np.ndarray, limits: np.ndarray, mutation_probability: int, distribution_index: int) -> None:
    n_population, n_var = population.shape
    # loop over the population
    for i in range(n_population):
        # loop over the variables
        for j in range(n_var):
            # validate the mutation probability
            if np.random.rand() <= mutation_probability:
                # get de individual
                individual = population[i][j]
                # generate a random number
                r = np.random.rand()                
                # get the lower and upper bounds
                lb = limits[j][0]
                ub = limits[j][1]
                # generate delta = min[ (ub(j)-population(i,j)), ( population(i,j) - lb(j) ) / (ub(j) - lb(j) ];
                delta = min( (ub - individual), (individual - lb) ) / (ub - lb)
                # validate the r <= 0.5
                if r <= 0.5:
                    # calculate delta_q = (2*r+(1-2*r)*(1-delta)^(Nm+1))^(1/(Nm+1))-1;
                    delta_q = (2*r + (1-2*r)*(1-delta)**(distribution_index+1))**(1/(distribution_index+1)) - 1
                else:
                    # calculate delta_q = 1-(2*(1-r)+2*(r-0.5)*(1-delta)^(Nm+1))^(1/(Nm+1));
                    delta_q = 1 - (2*(1-r) + 2*(r-0.5)*(1-delta)**(distribution_index+1))**(1/(distribution_index+1))
                
                #apply the mutation
                population[i][j] = individual + delta_q * (ub - lb)
            
                

if __name__ == '__main__':
    # JUST FOR TESTING
    # GENERAL CONFIGURATIONS
    limits = np.array([[1, 3], #limits var 1 -> [Lower, Upper]
                        [-1, 5]]) #limits var 2 -> [Lower, Upper]
    population = np.array([[1.127, -0.588]]) # just one individual for test

    # POLINOMIAL MUTATION CONFIGURATIONS
    mutation_probability_param = 10 #r -> Probability of mutation
    distribution_index_param = 20 #nm -> Distribution index ( ideal 20-100)
    
    # *hardcode the param r_1=0.4 and r_2=0.9 to get the same results as the class example
    
    #print the configuration and the population
    print(f'Limits: \n{limits}')
    print(f'Population: \n{population}')
    print(f'Mutation Probability: {mutation_probability_param}')
    print(f'Distribution Index: {distribution_index_param}')
    print(f'{"*"*50}')
    print(f'Applying Polinomial Mutation...')
    print(f'{"*"*50}')
    # apply the mutation
    apply_polinomial_mutation(population, limits, mutation_probability_param, distribution_index_param)    
    print(f'Population after mutation: \n{population}')
    
