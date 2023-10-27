import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def compute_spectrum(activations, images, labels,
                     dtype=torch.cuda.DoubleTensor, **kwargs):

    assert type(labels) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"

    Y = {key: lb.cuda().type(dtype) for key, lb in labels.items()}

    def get_spectral_properties(layer_act):

        # Compute
        # - Spectrum and weights
        # - Cumulative power
        # - Effective dimensions

        metric_keys = ['eigs',
                       'weights',
                       'orkhs_power',
                       'cum_powers',
                       'eff_dims',
                       'eff_dims_task',
                       'weights_Y',
                       'sample_size',
                       'feat_dim',
                       'neural_dim']

        metrics_uncent_cent = []

        for cent in [False, True]:

            (eig, weight,
             orkhs_power, weight_Y,
             P, N, neural_dim) = kernel_spectrum_from_feat(layer_act, Y=Y, cent=cent, **kwargs)

            eff_dim = eff_dimension(eig)
            cum_power, eff_dim_task = cum_powers(weight)

            metrics = [eig, weight, orkhs_power, cum_power,
                       eff_dim, eff_dim_task, weight_Y,
                       P, N, neural_dim]
            metrics = {key: val for key, val in zip(metric_keys, metrics)}
            metrics_uncent_cent += [metrics]

        return metrics_uncent_cent

    data_dict = dict(uncent={}, cent={})
    for layer_key, layer_act in tqdm(activations.items(), total=len(activations), desc='Layer'):

        layer_act = layer_act.cuda().type(dtype)
        P, N = layer_act.shape

        metrics_uncent, metrics_cent = get_spectral_properties(layer_act)

        data_dict['uncent'][layer_key] = metrics_uncent
        data_dict['cent'][layer_key] = metrics_cent

        layer_act = layer_act.cpu()
        torch.cuda.empty_cache()

    return data_dict


@torch.no_grad()
def cum_powers(weights, **kwargs):

    assert type(weights) is dict, "weights should be provided as a dict (e.g. {'classes': classes})"

    cum_powers = {}
    eff_dims_task = {}
    for key, weight in weights.items():
        cum_powers[key], eff_dims_task[key] = cum_power(weight)

    return cum_powers, eff_dims_task


@torch.no_grad()
def cum_power(weight):

    # Compute cumulative power for each task (label) and average
    weight_sq = weight**2
    cum_power = np.cumsum(weight_sq, axis=0) / weight_sq.sum(0)

    # Compute the effective dimension for each task
    eff_dim = np.array([eff_dimension(1-cum_power_task) for cum_power_task in cum_power.T])

    return cum_power.mean(-1), eff_dim.mean()


@torch.no_grad()
def kernel_spectrum_from_feat(feat, Y: dict, cent=False, **kwargs):

    assert type(Y) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"

    P, N = feat.shape
    for y in Y.values():
        assert y.shape[0] == P, "labels should have same sample size"

    # Center feats and targets
    if cent:
        feat = feat - feat.mean(0, keepdim=True)
        Y = {key: y - y.mean(0, keepdim=True) for key, y in Y.items()}

    # Compute the eigenspace of features and labels
    eigenspace_feat = get_eigenspace(feat, **kwargs)
    eigenspace_Y = {key: get_eigenspace(y, **kwargs) for key, y in Y.items()}

    # Feature eigenspace and its rank
    eig, vec, rank = [eigenspace_feat[k] for k in ['eig', 'vec', 'rank']]

    (weight, weight_Y, neural_dim) = {}, {}, {}
    for key, y, eigenspace_y in zip(Y.keys(), Y.values(), eigenspace_Y.values()):

        # Target eigenspace and its rank
        _, vec_y, _, C = [eigenspace_y[k] for k in ['eig', 'vec', 'rank', 'N']]

        # Weights of target projected on feature space and its own space
        weight_feat = vec.T @ y / P
        weight_tar = vec_y.T @ y / P

        # Store data
        weight[key] = weight_feat.cpu().numpy()
        weight_Y[key] = weight_tar.cpu().numpy()
        neural_dim[key] = C

    if rank < P:
        orkhs_power = {key: (w[rank:]**2).sum(0)/(w**2).sum(0) for key, w in weight.items()}
    else:
        orkhs_power = {key: np.zeros(w.shape[-1]) for key, w in weight.items()}

    return (eig.cpu().numpy(), weight,
            orkhs_power, weight_Y,
            P, N, neural_dim)


@torch.no_grad()
def eff_dimension(eig):
    return eig.sum()**2 / (eig**2).sum()


@torch.no_grad()
def get_eigenspace(act, to_numpy=False, threshold=1-1e-8, epsilon=1e-15, debug=False, **kwargs):

    if type(act) is np.ndarray:
        act = torch.from_numpy(act).cuda()
    assert act.dtype in ['float64', torch.float64]

    P, N = act.shape
    rank = min(P, N)

    # Compute kernel PCA
    K = act @ act.T
    Id = torch.eye(P, device=K.device, dtype=K.dtype)

    try:
        eig, vec = torch.linalg.eigh(K/P + epsilon*Id)
    except Exception as e:
        print(e, 'Trying eigh with higher regularization')
        K = act @ act.T
        eig, vec = torch.linalg.eigh(K/P + 1e-8*Id)
    eig = torch.flip(eig, dims=[0])
    vec = torch.flip(vec, dims=[1]) * np.sqrt(P)

    return_names = ['eig', 'vec',
                    'P', 'N', 'rank']

    returns = (eig, vec,
               P, N, rank)

    return_dict = {key: val for key, val in zip(return_names, returns)}

    if to_numpy:
        for key, val in return_dict.items():
            return_dict[key] = val.cpu().numpy() if torch.is_tensor(val) else val
    return return_dict
