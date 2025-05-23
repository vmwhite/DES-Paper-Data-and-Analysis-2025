from scipy.stats import qmc


def gen_matrix(num):
    seed_num = 68432
    sampler = qmc.Sobol(d=41, seed = seed_num) #generate for 41 seperate parameters
    sample = sampler.random_base2(m=10) # 2^m or if m=1024 Sobol sets
    # sample = sampler.random(512) #use to specify specific number of sets to generate, not recommended
    # print(sample)

    # translate Sobol Sample to Paramerter lower and upper bounds
    l_bounds = [ 1.98, 0.72, 3.55, 0.47, 25934.05, 4.75, 14.25, 285.00, 0.19, 10.32, 16.72, 3.98, 16.28, 3.93, 8.62, 2.86, 9.54, 2.24, 7.10, 0.98, 4.58, 2.09, 7.49, 2.26, 0.0095, 0.2116, 0.0207, 0.78, 0.45, 2.05, 1.40, 4.54, 1.12, 3.12, 1.53, 4.30, 1.04, 1.86, 1.33, 5.98, 2.38]
    u_bounds = [2.18, 0.80, 3.93, 0.51, 46291.35, 15.75, 52.50, 525.00, 0.84, 11.41, 18.47, 4.40, 17.99, 4.35, 9.53, 3.16, 10.55, 2.47, 7.85, 1.08, 5.06, 2.31, 8.27, 2.50, 0.0105, 0.2338, 0.0229, 0.86, 0.50, 2.27, 1.54, 5.02, 1.23, 3.45, 1.69, 4.75, 1.15, 2.05, 1.47, 6.61, 2.63]

    matrix = qmc.scale(sample, l_bounds, u_bounds)
    # print(matrix)

    # Calcualte the quality of the samle using discrprency
    print("---- Sobol Sample Discrepancy ----")
    print("CD Discrepency = ", qmc.discrepancy(sample, method="CD")) #the lower the better
    print("WD Discrepency = ", qmc.discrepancy(sample, method="WD")) #the lower the better
    print("MD Discrepency = ", qmc.discrepancy(sample, method="MD")) #the lower the better
    print("L2-star Discrepency = ", qmc.discrepancy(sample, method="L2-star")) #the lower the better

    return matrix


'''
    From: https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_qmc.py#L1284-L1596 
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.
    The lower the value is, the better the coverage of the parameter space is.
    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    Four methods are available:
    * ``CD``: Centered Discrepancy - subspace involves a corner of the
      hypercube
    * ``WD``: Wrap-around Discrepancy - subspace can wrap around bounds
    * ``MD``: Mixture Discrepancy - mix between CD/WD covering more criteria
    * ``L2-star``: L2-star discrepancy - like CD BUT variant to rotation
    See [2]_ for precise definitions of each method.

    References
    ----------
    .. [1] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [2] Zhou Y.-D. et al. "Mixture discrepancy for quasi-random point sets."
       Journal of Complexity, 29 (3-4) , pp. 283-301, 2013.
    .. [3] T. T. Warnock. "Computational investigations of low discrepancy
       point sets." Applications of Number Theory to Numerical
       Analysis, Academic Press, pp. 319-343, 1972.
    '''
'''
#For printing total list of paramters to csv
import os
import pandas as pd
cwd = os.getcwd() 
matrix = gen_matrix(1024)
pd.DataFrame(matrix).to_csv('Revison2_Results\AppD_Sensitivity\Sens_Params.csv')
'''
