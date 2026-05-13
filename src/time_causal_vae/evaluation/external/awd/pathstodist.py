import numpy as np
from time_causal_vae.evaluation.external.awd.gan.adaptedempirical import (
    empirical_k_means_measure,
    empirical_k_means_measure_markov,
)
from time_causal_vae.evaluation.external.awd.measures import get_full_disintegration, Graph
from time_causal_vae.evaluation.external.awd.mainfunctions import solve_dynamic
from concurrent.futures import ProcessPoolExecutor

COST_P_PAR = 1


def f_lp_parallel(x, y):
    return np.sum(np.abs(x - y)) ** COST_P_PAR


def eval_calc_aw(input_tuple):
    return calc_one_aw(*input_tuple)


def calc_one_aw(
    inds,
    paths1h,
    paths2h,
    T1,
    use_klist,
    k_list,
    verbose,
    markov,
    emp_k_means_fun,
    get_full_dis_fun,
    emp_k_means_markov_fun,
    solve_dynamic_fun,
):
    tmh = len(inds)
    g = Graph(tmh)
    if use_klist:
        klisth = []
        for i in range(T1):
            if i in inds:
                klisth.append(k_list[i])
    else:
        klisth = k_list
    if not markov:
        for i in range(tmh - 1):
            for j in range(i + 1, tmh):
                g.addEdge(i, j)
        if verbose:
            print("Computing adapted empirical measures...")
        x1a, w1a = emp_k_means_fun(paths1h, klist=klisth, use_klist=use_klist)
        x2a, w2a = emp_k_means_fun(paths2h, klist=klisth, use_klist=use_klist)
        if verbose:
            print("Done!")
            print("Getting disintegration...")
        mu, supp_mu = get_full_dis_fun([x1a, w1a], g, list(range(tmh)))
        mu2, supp_mu2 = get_full_dis_fun([x2a, w2a], g, list(range(tmh)))
        if verbose:
            print("Done!")
    else:
        for i in range(tmh - 1):
            g.addEdge(i, i + 1)
        if verbose:
            print("Computing adapted empirical maps (Markovian)")
        mu, supp_mu = emp_k_means_markov_fun(paths1h, klist=klisth, use_klist=use_klist)
        mu2, supp_mu2 = emp_k_means_markov_fun(paths2h, klist=klisth, use_klist=use_klist)
        if verbose:
            print("Done!")

    if verbose:
        print("Solving adapted OT problem using backward induction...")
    cost_funs = [[[t], f_lp_parallel] for t in range(tmh)]

    v, pi_opt = solve_dynamic_fun(cost_funs, mu, mu2, supp_mu, supp_mu2, g, method="pot", outputflag=verbose)
    return v[0]


def paths_to_dist_parallel(
    paths1,
    paths2,
    n_slices=1,
    len_slices=0,
    use_klist=0,
    k_list=(),
    markov=0,
    verbose=0,
    max_workers=4,
):
    # if len_slices = 0, then the whole paths are used (i.e., all time steps)
    # Note that there are several sources of randomness in this algorithm, like:
    # 1) randomness of solution of Kmeans algorithm for adapted empirical map
    # 2) if n_slices>1, then slices of time are chosen randomly

    n1, T1 = paths1.shape
    n2, T2 = paths2.shape
    assert T1 == T2

    if len_slices == 0 or len_slices == T1:
        n_slices = 1  # makes no sense to use several runs if we use all time steps anyways

    if len_slices == 0:
        len_slices = T1

    inputs_calc = []
    for i in range(
        n_slices
    ):  # inds, paths1h, paths2h, T1, use_klist, k_list, verbose, markov, emp_k_means_fun, get_full_dis_fun, emp_k_means_markov_fun, solve_dynamic_fun
        inds = np.random.choice(list(range(T1)), size=len_slices, replace=False)
        inds = np.sort(inds)
        paths1h = paths1[:, inds]
        paths2h = paths2[:, inds]
        inph = (
            inds,
            paths1h,
            paths2h,
            T1,
            use_klist,
            k_list,
            verbose,
            markov,
            empirical_k_means_measure,
            get_full_disintegration,
            empirical_k_means_measure_markov,
            solve_dynamic,
        )
        inputs_calc.append(inph)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        out_dists = list(executor.map(eval_calc_aw, inputs_calc))
    if verbose:
        print(out_dists)
    dist_val = np.mean(out_dists)
    return dist_val
