import numpy as np
from sklearn.cluster import KMeans
from time_causal_vae.evaluation.external.awd.measures import Graph, get_full_disintegration, combine_markov


def empirical_k_means_measure(data, use_klist=0, klist=(), tol_decimals=6, use_weights=1, heuristic=0):
    """

    :param data: np array of size [k, T_h], where k is number of samples and T_h is number of time steps
    :param use_klist: 0 or 1, whether to use a predefined partition in each time step or not
    :param klist: only used if use_klist=1. Then, has to be a list of size T_h, where klist[t] specifies how many
                  support points the adapted empirical measure will have at time t
    :param tol_decimals: integer, rounding precision
    :param use_weights: should be 1, otherwise just all cluster centers are equally weighed which leads to weird results
    :param heuristic: should be 0, otherwise it employs a heuristic clustering instead of kmeans
    :return: out_x, out_w ; where out_x is np array of size [n, T_h] (support of quantized measure) and out_w of size n
    """
    # data is [k, T_h] array
    # klist is list with T_h entries, each being an integer lower than k; number of barycenters for each time step
    (k, T_h) = data.shape
    if not use_klist:
        klist = (np.ones(T_h) * int(np.round(np.sqrt(k)))).astype(int)

    label_list = []
    support_list = []
    out_x = np.zeros([0, T_h])
    out_w = []

    # cluster points at each time point
    # print('Clustering...')
    if heuristic:
        for t in range(T_h):
            data_t = data[:, t]
            inds_sort_t = np.argsort(data_t)
            datas_t = data_t[inds_sort_t]
            n_av = int(np.round(k / klist[t]))
            lmax = int(np.floor(n_av * klist[t]))
            all_but_end = np.reshape(datas_t[:lmax], (-1, n_av))
            mean_all_but = np.mean(all_but_end, axis=1, keepdims=1)
            cx = mean_all_but
            mean_all_but = np.tile(mean_all_but, (1, n_av))
            mean_all_but = np.reshape(mean_all_but, (-1, 1))
            mean_rest = np.mean(datas_t[lmax:])
            if lmax < k:
                mean_vec = np.concatenate([np.squeeze(mean_all_but), np.array([mean_rest])])
                cx = np.concatenate([cx, np.array([mean_rest])])
            else:
                mean_vec = np.squeeze(mean_all_but)
            lx = np.zeros(k, dtype=int)
            for i in range(k):
                for j in range(len(cx)):
                    if mean_vec[inds_sort_t[i]] == cx[j]:
                        lx[i] = j
                        continue
            label_list.append(lx)
            support_list.append(cx)

    else:
        for t in range(T_h):
            # print('t = ' + str(t))
            data_t = data[:, t : t + 1]
            kmx = KMeans(n_clusters=klist[t]).fit(data_t)

            cx = kmx.cluster_centers_
            cx = np.round(cx, decimals=tol_decimals)
            lx = kmx.labels_
            label_list.append(lx)
            support_list.append(cx)

    if use_weights == 0:  # weight all cluster centers equally? ... Convenient but theoretically flawed I think
        out = np.zeros([k, T_h])
        for t in range(T_h):
            out[:, t] = support_list[t][label_list[t]][:, 0]
        return out

    # build output measure
    for i in range(k):
        cur_path = np.zeros(T_h)
        for t in range(T_h):
            cur_path[t] = support_list[t][label_list[t][i]]

        # check whether the path already exists
        path_is_here = 0
        for j in range(len(out_w)):
            if np.all(out_x[j, :] == cur_path):
                out_w[j] += 1 / k
                path_is_here = 1
                break
        if not path_is_here:
            out_x = np.append(out_x, np.expand_dims(cur_path, axis=0), axis=0)
            out_w.append(1 / k)

    return out_x, out_w


def empirical_k_means_measure_markov(data, use_klist=0, klist=(), tol_decimals=6, use_weights=1, heuristic=0):
    """

    :param data: np array of size [k, T_h], where k is number of samples and T_h is number of time steps
    :param use_klist: 0 or 1, whether to use a predefined partition in each time step or not
    :param klist: only used if use_klist=1. Then, has to be a list of size T_h, where klist[t] specifies how many
                  support points the adapted empirical measure will have at time t
    :param tol_decimals: integer, rounding precision
    :param use_weights: should be 1, otherwise just all cluster centers are equally weighed which leads to weird results
    :param heuristic: should be 0, otherwise it employs a heuristic clustering instead of kmeans
    :return: out_x, out_w ; where out_x is np array of size [n, T_h] (support of quantized measure) and out_w of size n
    """
    # data is [k, T_h] array
    # klist is list with T_h entries, each being an integer lower than k; number of barycenters for each time step
    (k, T_h) = data.shape
    if not use_klist:
        klist = (np.ones(T_h) * int(np.round(np.sqrt(k)))).astype(int)

    label_list = []
    support_list = []
    out_x = np.zeros([0, T_h])
    out_w = []

    # cluster points at each time point
    # print('Clustering...')
    if heuristic:
        for t in range(T_h):
            data_t = data[:, t]
            inds_sort_t = np.argsort(data_t)
            datas_t = data_t[inds_sort_t]
            n_av = int(np.round(k / klist[t]))
            lmax = int(np.floor(n_av * klist[t]))
            all_but_end = np.reshape(datas_t[:lmax], (-1, n_av))
            mean_all_but = np.mean(all_but_end, axis=1, keepdims=1)
            cx = mean_all_but
            mean_all_but = np.tile(mean_all_but, (1, n_av))
            mean_all_but = np.reshape(mean_all_but, (-1, 1))
            mean_rest = np.mean(datas_t[lmax:])
            if lmax < k:
                mean_vec = np.concatenate([np.squeeze(mean_all_but), np.array([mean_rest])])
                cx = np.concatenate([cx, np.array([mean_rest])])
            else:
                mean_vec = np.squeeze(mean_all_but)
            lx = np.zeros(k, dtype=int)
            for i in range(k):
                for j in range(len(cx)):
                    if mean_vec[inds_sort_t[i]] == cx[j]:
                        lx[i] = j
                        continue
            label_list.append(lx)
            support_list.append(cx)

    else:
        for t in range(T_h):
            # print('t = ' + str(t))
            data_t = data[:, t : t + 1]
            kmx = KMeans(n_clusters=klist[t]).fit(data_t)
            cx = kmx.cluster_centers_
            cx = np.round(cx, decimals=tol_decimals)
            lx = kmx.labels_
            label_list.append(lx)
            support_list.append(cx)

    if use_weights == 0:  # weight all cluster centers equally? ... Convenient but theoretically flawed I think
        out = np.zeros([k, T_h])
        for t in range(T_h):
            out[:, t] = support_list[t][label_list[t]][:, 0]
        return out

    # build output measure
    mu_list = []
    for t_build in range(T_h - 1):
        out_x = np.zeros([0, 2])
        out_w = []
        for i in range(k):
            cur_path = np.zeros(2)
            for t in range(t_build, t_build + 2):
                cur_path[t - t_build] = support_list[t][label_list[t][i]]
            path_is_here = 0
            for j in range(len(out_w)):
                if np.all(out_x[j, :] == cur_path):
                    out_w[j] += 1 / k
                    path_is_here = 1
                    break
            if not path_is_here:
                out_x = np.append(out_x, np.expand_dims(cur_path, axis=0), axis=0)
                out_w.append(1 / k)
        g = Graph(2)
        g.addEdge(t_build, t_build + 1)
        mu_dis, supp = get_full_disintegration([out_x, out_w], g, [t_build, t_build + 1])
        mu_list.append([mu_dis, supp])
    return combine_markov(mu_list)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.mainfunctions import solve_dynamic

    N_DATA = 500
    T_MAX = 5
    reals1 = np.load("reals1.npy")
    reals1 = reals1[:N_DATA, :T_MAX]
    reals2 = np.load("reals2.npy")
    reals2 = reals2[:N_DATA, :T_MAX]
    _, T = reals1.shape
    k_list = [1] + [int(np.round(N_DATA ** (1 / 2))) for t in range(1, T)]
    mu, supp_mu = empirical_k_means_measure_markov(reals1, klist=k_list, use_klist=1)
    mu2, supp_mu2 = empirical_k_means_measure_markov(reals2, klist=k_list, use_klist=1)

    # N_SIM = 100
    # path_test = np.zeros([N_SIM, T])
    # for i in range(N_SIM):
    #     path_test[i, 0] = np.random.choice(mu(0, [])[0].flatten(), p=mu(0, [])[1])
    #     for t in range(1, T):
    #         path_test[i, t] = np.random.choice(mu(t, [path_test[i, t-1]])[0].flatten(), p=mu(t, [path_test[i, t-1]])[1])
    #
    # for i in range(N_DATA):
    #     plt.plot(reals1[i, :])
    # plt.show()
    # for i in range(N_SIM):
    #     plt.plot(path_test[i, :])
    # plt.show()
    g = Graph(T)
    for t in range(T - 1):
        g.addEdge(t, t + 1)

    def f_lp(x, y):
        return np.sum(np.abs(x - y))

    cost_funs = [[[t], f_lp] for t in range(T)]
    v, _ = solve_dynamic(cost_funs, mu, mu2, supp_mu, supp_mu2, g, method="pot", outputflag=0)
    print(v[0])

    x1a, w1a = empirical_k_means_measure(reals1, klist=k_list, use_klist=1)
    x2a, w2a = empirical_k_means_measure(reals2, klist=k_list, use_klist=1)
    mu_1_dis, supp1 = get_full_disintegration([x1a, w1a], g, list(range(T)))
    mu_2_dis, supp2 = get_full_disintegration([x2a, w2a], g, list(range(T)))
    v, _ = solve_dynamic(cost_funs, mu_1_dis, mu_2_dis, supp1, supp2, g, method="pot", outputflag=0)
    print(v[0])
    # END OF MARKOV TEST #

    exit()
    import matplotlib.pyplot as plt

    N_DATA = 100
    T_MAX = 5
    reals1 = np.load("reals1.npy")
    reals2 = np.load("reals2.npy")
    fakes = np.load("fakes.npy")
    fakes -= fakes[:, 0:1]
    reals1 = reals1[:N_DATA, :T_MAX]
    reals2 = reals2[:N_DATA, :T_MAX]
    fakes = fakes[:N_DATA, :T_MAX]
    n1 = len(reals1)
    n2 = len(reals2)
    nf = len(fakes)
    w1 = np.ones(n1) / n1
    w2 = np.ones(n2) / n2
    wf = np.ones(nf) / nf
    _, T = reals1.shape

    mu_1 = [reals1, w1]
    mu_2 = [reals2, w2]
    mu_3 = [fakes, wf]
    control_mu = [np.zeros([1, T]), np.ones(1)]

    # k_list = [1] + [int(np.round(N_DATA ** (1 / (T - 1)))) for t in range(1, T)]
    k_list = [1] + [int(np.round(N_DATA ** (1 / 2))) for t in range(1, T)]

    print(k_list)
    for i in range(N_DATA):
        plt.plot(reals1[i, :])
    plt.title("Raw empirical measure")
    plt.show()
    x1a, w1a = empirical_k_means_measure(reals1, klist=k_list, use_klist=1)
    for i in range(len(w1a)):
        plt.plot(x1a[i, :], linewidth=np.sqrt(N_DATA) * np.sqrt(w1a[i]))
    plt.title("Adapted empirical measure")
    plt.show()
