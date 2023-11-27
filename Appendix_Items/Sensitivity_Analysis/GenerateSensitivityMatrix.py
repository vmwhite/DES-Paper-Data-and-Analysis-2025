from scipy.stats import qmc


def gen_matrix(num):
    seed_num = 68432
    sampler = qmc.Sobol(d=39, seed = seed_num) #generate for 37 seperate parameters
    sample = sampler.random_base2(m=9) # 2^m or if m=9 512 Sobol sets
    # sample = sampler.random(512) #use to specify specific number of sets to generate, not recommended
    # print(sample)

    # translate Sobol Sample to Paramerter lower and upper bounds
    l_bounds = [2.733424247,0.5309296,2.948082891,0.438312942,25934.05,4.75,14.25,285,0.19,10.322605,16.7154191,3.98492795,16.276502,3.9322609,0.97711,8.61911801,2.861495,9.541468935,2.236809618,7.1032488,0.96479245,0.0095,0.211565,4.5794883,2.08578865,0.775147514,0.453776017,2.05171078,1.396110755,4.543247561,1.11646968,3.123040133,1.530138162,4.296709354,1.036609529,1.855200335,1.327569327,5.9771834,2.382923] # list lower bounds here
    u_bounds = [3.021153115,0.586816926,3.258407406,0.484451146,46291.35,15.75,52.5,525,0.84,11.409195,18.4749369,4.40439405,17.989818,4.3461831,0.97929,9.52639359,3.162705,10.54583409,2.472263262,7.8509592,1.06634955,0.0105,0.233835,5.0615397,2.30534535,0.856741989,0.501541913,2.267680336,1.543069782,5.021484146,1.233992804,3.451781199,1.691205337,4.74899455,1.145726322,2.050484581,1.467313467,6.6063606,2.633757]
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