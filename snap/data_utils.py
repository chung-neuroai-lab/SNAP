import warnings

import os
import numpy as np
import pandas as pd


def create_dataframe(model_dict, identifier, centered=True):

    # identifier = 'V1_cornet_s'
    temp = model_dict.copy()
    layers = temp.pop('layers')

    final_df = None
    for metric_key, metric_dict in temp.items():

        metric_dict = metric_dict['cent'] if centered else metric_dict['uncent']

        # Initialize concatenating dictionary
        concat_layers = {key: [] for key in metric_dict[layers[0]].keys()}
        for key, val in metric_dict[layers[0]].items():
            if isinstance(val, dict):
                concat_layers[key] = {}
                for key_val, item in val.items():
                    if isinstance(item, dict):
                        print('here')
                        concat_layers[key][key_val] = {item_key: [] for item_key in item.keys()}
                    else:
                        concat_layers[key][key_val] = []
            else:
                concat_layers[key] = []

        # Concatenating items from each layer
        for layer in layers:
            for key, val in metric_dict[layer].items():
                if isinstance(val, dict):
                    for key_val, item in val.items():
                        if isinstance(item, dict):
                            print('here')
                            concat_layers[key][key_val] = {item_key: concat_layers[key]
                                                           [key_val] + [item1] for item_key, item1 in item.items()}
                        else:
                            concat_layers[key][key_val] = concat_layers[key][key_val] + [item]
                else:
                    concat_layers[key] = concat_layers[key] + [metric_dict[layer][key]]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Make them numpy arrays
            temp = {}
            for key, val in concat_layers.items():
                if isinstance(val, dict):
                    for key_val, item in val.items():
                        if isinstance(item, dict):
                            for key_val1, item1 in item.items():
                                temp[key+'_'+key_val+'_'+key_val1] = [np.array(item1)]
                        else:
                            np.array(item)

                            if len(w) != 0:
                                temp[key+'_'+key_val] = [item]
                            else:
                                temp[key+'_'+key_val] = [np.array(item)]

                else:
                    temp[key] = [np.array(val)]

        concat_layers = temp

        df = pd.DataFrame.from_dict(concat_layers, orient='index').transpose()

        if final_df is None:
            final_df = df
        else:
            final_df = final_df.join(df)

    identifier_columns = ["pooling", "region", "model", "trained"]

    final_df.rename(index={0: tuple(identifier)}, inplace=True)

    final_df['layers'] = [layers]
    for key, name in zip(identifier_columns, identifier):
        final_df[key] = [name]

    index = pd.MultiIndex.from_frame(
        final_df[["pooling", "region", "model", "trained"]],
        names=["pooling", "region", "model", "trained"])
    final_df = final_df.reindex(index)
    final_df.drop(columns=["pooling", "region", "model", "trained"], inplace=True)

    return final_df


def gather_data(data_root, activation_pooling, regionNames, modelNames, pretrained, load=True, save_all_data_pckl=True):

    dfs_all = []
    for pooling in activation_pooling:

        data_dir = data_root + f"data_{pooling}"

        if os.path.exists(data_dir):

            if os.path.isfile(data_dir + "/all_data.pkl") and load:
                print('Loading dataframe')
                dfs = pd.read_pickle(data_dir + "/all_data.pkl")
            else:
                print('Generating dataframe')
                dfs = []
                for region in regionNames:
                    for model_name in modelNames:
                        for trained in [True, False]:
                            try:
                                identifier = [pooling, region, model_name, pretrained[trained]]
                                data_fname = data_dir + f"/{region}_data_{model_name}_{pretrained[trained]}.npz"
                                data = np.load(data_fname, allow_pickle=True)['exp_metrics'].tolist()
                                dfs += [create_dataframe(data, identifier)]
                            except Exception as e:
                                print(f'Ignoring data {data_fname}', e)
                                pass

                dfs = pd.concat(dfs)
                if save_all_data_pckl:
                    dfs.to_pickle(data_dir + "/all_data.pkl")

            dfs_all += [dfs]

        else:
            print(f'cant find {data_dir}')

    dfs_all = pd.concat(dfs_all)
    print('Done!')

    return dfs_all


class DataProcess:

    def __init__(self, data_root, pooling_list, region_list, model_list, pretrained_dict):

        self.data_root = data_root
        self.pooling_list = pooling_list
        self.region_list = region_list
        self.model_list = model_list
        self.pretrained_dict = pretrained_dict

    def get_dataframe(self, load=True, save_all_data_pckl=False):

        self.dfs_all = gather_data(self.data_root, self.pooling_list, self.region_list,
                                   self.model_list, self.pretrained_dict, load=load,
                                   save_all_data_pckl=save_all_data_pckl)

        return self.dfs_all

    def query(self, pooling, region, trained, model_list=None):

        query = f'pooling == "{pooling}" and region == "{region}" and trained == "{self.pretrained_dict[trained]}"'
        if model_list is not None:
            query += f' and model in {model_list}'

        return self.dfs_all.query(query)

    def process_model(self, pooling, region, trained, model_list=None, eff_dim_cutoff=0, threshold=0.9999):

        df = self.query(pooling, region, trained, model_list=model_list)
        return process_model(df, eff_dim_cutoff=eff_dim_cutoff, threshold=threshold)

    def get_all_data(self, region_list=None, pooling_list=None, model_list=None, **kwargs):

        pooling_list = self.pooling_list if pooling_list is None else pooling_list
        region_list = self.region_list if region_list is None else region_list
        model_list = self.model_list if model_list is None else model_list

        kwargs |= dict(pooling_list=pooling_list,
                       region_list=region_list,
                       model_list=model_list,)

        return get_all_data(process_fn=self.process_model, **kwargs)


def eff_dimension(eig):

    if len(eig.shape) == 1:
        return eig.sum()**2 / (eig**2).sum()

    return eig.sum(1)**2 / (eig**2).sum(1)


def mode_threshold(eig, threshold):
    cumsum = eig.cumsum(0) / eig.sum()
    idx = np.where(cumsum > threshold)[0]
    if len(idx) == 0:
        rank = len(eig)
    else:
        rank = idx[0] + 1
    return rank


def process_model(df, eff_dim_cutoff=0, threshold=0.9999):

    model_reg_dict = {}
    for model, model_data in df.iterrows():

        layers = model_data.layers[:-2]
        models = np.array([model[2]]*len(layers), dtype=str)

        # Experiment curves #
        pvals = model_data.responses_pvals[:-2]
        P = model_data.responses_P[:-2]
        N = model_data.responses_N[:-2]
        C = model_data.responses_C[:-2]

        # Mean over trials - Sum over neurons/classes
        gen_errs = model_data.responses_gen_errs[:-2]
        tr_errs = model_data.responses_tr_errs[:-2]
        test_errs = model_data.responses_test_errs[:-2]
        gen_norm = model_data.responses_gen_norm[:-2]
        tr_norm = model_data.responses_tr_norm[:-2]
        test_norm = model_data.responses_test_norm[:-2]
        assert np.allclose(P[0]*gen_errs*gen_norm, np.einsum('ij,ikjl->ikjl', pvals,
                           tr_errs*tr_norm) + np.einsum('ij,ikjl->ikjl', P[0]-pvals, test_errs*test_norm))

        # Mean over trials - Sum over neurons/classes
        gen_errs = gen_errs.mean(1).sum(-1)
        tr_errs = tr_errs.mean(1).sum(-1)
        test_errs = test_errs.mean(1).sum(-1)
        gen_norm = gen_norm.mean(1).sum(-1)
        tr_norm = tr_norm.mean(1).sum(-1)
        test_norm = test_norm.mean(1).sum(-1)

        pearson_gen = model_data.responses_pearson_gen[:-2].mean(1).mean(-1)
        pearson_tr = model_data.responses_pearson_tr[:-2].mean(1).mean(-1)
        pearson_test = model_data.responses_pearson_test[:-2].mean(1).mean(-1)

        # Theory curves #
        pvals_theory = model_data.responses_pvals_theory[:-2]
        # Sum over neurons/classes
        gen_theory = model_data.responses_gen_theory[:-2].sum(-1)
        tr_theory = model_data.responses_tr_theory[:-2].sum(-1)
        mode_errs = model_data.responses_mode_err_theory[:-2]
        eff_regs = model_data.responses_eff_regs[:-2]

        # Min Gen Error, Max Train Error #
        min_gen_errs = gen_errs[:, :-1].min(-1)
        max_tr_errs = tr_errs[:, -1]

        # Spectral Data #
        eigs = model_data.eigs[:-2][:, eff_dim_cutoff:]
        weights = model_data.weights_responses[:-2][:, eff_dim_cutoff:]
        weights_task = model_data.weights_Y_responses[:-2][:, eff_dim_cutoff:]
        orkhs_powers = model_data.orkhs_power_responses[:-2][:, eff_dim_cutoff:].sum(-1)

        weights_sq = np.einsum('pri, pri->pr', weights, weights)
        weights_task_sq = np.einsum('pri, pri->pr', weights_task, weights_task)

        # Normalize spectrum to unit trace
        weights = weights / np.sqrt(weights_sq.sum(-1))[:, None, None]
        weights_task = weights_task / np.sqrt(weights_task_sq.sum(-1))[:, None, None]

        eigs_sum = eigs.sum(-1)
        weights_sq_sum = weights_sq.sum(-1)
        weights_task_sq_sum = weights_task_sq.sum(-1)

        eigs = eigs / eigs_sum[:, None]
        weights_sq = weights_sq / weights_sq_sum[:, None]
        weights_task_sq = weights_task_sq / weights_task_sq_sum[:, None]

        assert np.allclose(weights_sq.sum(-1), 1)
        assert np.allclose(weights_sq, (weights**2).sum(-1)), 'pass'
        assert np.allclose((weights_task**2).sum(-1), weights_task_sq), 'pass'
        assert np.allclose(eigs.sum(-1), weights_sq.sum(-1))
        assert np.allclose(weights_task_sq.sum(-1), weights_sq.sum(-1))

        # Compute effective dimensions
        cum_powers_eig = np.cumsum(eigs, axis=1) / eigs.sum(1, keepdims=True)
        cum_powers_weight = np.cumsum(weights_sq, axis=1) / weights_sq.sum(1, keepdims=True)
        cum_powers_task = np.cumsum(weights_task_sq, axis=1) / weights_task_sq.sum(1, keepdims=True)

        eds = eff_dimension(eigs)
        cum_eds = eff_dimension(1-cum_powers_eig)

        tads = eff_dimension(weights_sq)
        cum_tads = eff_dimension(1-cum_powers_weight)

        task_eds = eff_dimension(weights_task_sq)
        cum_task_eds = eff_dimension(1-cum_powers_task)

        # Alternative effective dimensions
        feat_dims = eff_dimension(eigs * (1 - cum_powers_eig))
        task_dims = eff_dimension(weights_sq * (1 - cum_powers_weight))

        # Compute final regression score
        final_scores = []
        ignore_idx = []
        for idx, (orkhs, gen, tr) in enumerate(zip(orkhs_powers, min_gen_errs, max_tr_errs)):

            if tr > orkhs and tr < 1:
                alternative = tr
            else:
                alternative = orkhs

            if (gen_errs[idx] > 1).sum() > 0 and gen_errs[idx, -2] > gen:
                if alternative < 1e-1:
                    final_scores += [gen]
                    ignore_idx += [idx]
                else:
                    final_scores += [alternative]
            else:
                final_scores += [gen]
        final_scores = np.array(final_scores)

        dict_keys = ['models', 'layers',
                     'eigs', 'weights', 'weights_sq',
                     'cum_powers_eig', 'cum_powers_weight',
                     'orkhs_powers', 'weights_task_sq',
                     'eigs_sum', 'weights_sq_sum', 'weights_task_sq_sum',

                     'eds', 'cum_eds',
                     'tads', 'cum_tads',
                     'task_eds', 'cum_task_eds',
                     'feat_dims', 'task_dims',

                     'pvals', 'P', 'N', 'C',
                     'gen_errs', 'tr_errs', 'test_errs',
                     'gen_norm', 'tr_norm', 'test_norm',
                     'pearson_gen', 'pearson_tr', 'pearson_test',

                     'pvals_theory', 'gen_theory', 'tr_theory',
                     'mode_errs', 'eff_regs',

                     'min_gen_errs', 'max_tr_errs',

                     'final_scores',
                     ]

        dict_vals = (models, layers,
                     eigs, weights, weights_sq,
                     cum_powers_eig, cum_powers_weight,
                     orkhs_powers, weights_task_sq,
                     eigs_sum, weights_sq_sum, weights_task_sq_sum,

                     eds, cum_eds,
                     tads, cum_tads,
                     task_eds, cum_task_eds,
                     feat_dims, task_dims,

                     pvals, P, N, C,
                     gen_errs, tr_errs, test_errs,
                     gen_norm, tr_norm, test_norm,
                     pearson_gen, pearson_tr, pearson_test,

                     pvals_theory, gen_theory, tr_theory,
                     mode_errs, eff_regs,

                     min_gen_errs, max_tr_errs,

                     final_scores,
                     )

        model_reg_dict[model[2]] = {key: val for key, val in zip(dict_keys, dict_vals)}
        model_reg_dict[model[2]] |= compute_mode_errs(eigs, weights, pvals,
                                                      threshold, reg=1e-14)

    return model_reg_dict


def get_all_data(process_fn, sort_coord, trained, region_list, pooling_list,
                 model_list, eff_dim_cutoff, threshold):

    all_reg_hist = {}
    all_processed_data = {}
    for region in region_list:
        processed_data = {}
        for pooling in pooling_list:
            processed_data[pooling] = process_fn(pooling, region,
                                                 trained, model_list=model_list,
                                                 eff_dim_cutoff=eff_dim_cutoff,
                                                 threshold=threshold)

        # Create a dict of data for all models
        reg_models_hist = {}
        for pooling in pooling_list:
            pooling_data = processed_data[pooling]
            models_data_keys = ['models', 'layers',
                                'ranks_eig', 'ranks_weight',
                                'eds', 'cum_eds',
                                'tads', 'cum_tads',
                                'task_eds', 'cum_task_eds',
                                'feat_dims', 'task_dims',
                                'min_gen_errs', 'max_tr_errs',
                                'final_scores',

                                'pvals_mode',
                                'dyn_tads', 'dyn_eds', 'dyn_null_eds',
                                'dyn_weight_rads', 'dyn_eig_rads', 'dyn_null_rads',]
            models_data = {key: [] for key in models_data_keys}

            # Collect all data
            for data_key, data in pooling_data.items():
                for key, val in models_data.items():
                    try:
                        assert data[key].ndim == 1, f'{key} is not 1 D'
                    except Exception:
                        # print(data[key])
                        pass
                    models_data[key] = val + [data[key]]

            # Collect concatenate data
            for key, val in models_data.items():
                models_data[key] = np.concatenate(list(val))

            # Sort data
            sort_idx = np.argsort(models_data[sort_coord])[::-1]
            for key, val in models_data.items():
                models_data[key] = val[sort_idx]

            reg_models_hist[pooling] = models_data
        all_reg_hist[region] = reg_models_hist
        all_processed_data[region] = processed_data

    return all_reg_hist, all_processed_data


def compute_mode_errs(eigs, weights, pvals, threshold=0.99, reg=1e-14):
    import snap.regression_utils as reg_utils

    if len(eigs.shape) == 1:
        eigs, weights, pvals = [val[None, :] for val in [eigs, weights, pvals]]

    return_keys = ['pvals_mode', 'mode_errs',
                   'dyn_weights', 'dyn_eigs', 'dyn_nulls',
                   'dyn_tads', 'dyn_eds', 'dyn_null_eds',
                   'dyn_weight_rads', 'dyn_eig_rads', 'dyn_null_rads',
                   'ranks_eig', 'ranks_weight']
    returns = {key: [] for key in return_keys}

    for eig, weight, pval in zip(eigs, weights, pvals):

        theory = reg_utils.gen_error_theory(eig, weight, reg, pval)
        mode_err = theory['mode_err_theory']

        weight_sq = (weight**2).sum(-1)
        dyn_weight = mode_err * weight_sq[None, :]
        dyn_eig = mode_err * eig[None, :]
        dyn_null = mode_err

        assert np.allclose(eig.sum(), 1) and np.allclose(weight_sq.sum(), 1)
        assert np.allclose(theory['gen_theory'].sum(-1), dyn_weight.sum(-1))

        returns['pvals_mode'] += [pval]
        returns['mode_errs'] += [mode_err]

        returns['dyn_weights'] += [dyn_weight]
        returns['dyn_eigs'] += [dyn_eig]
        returns['dyn_nulls'] += [dyn_null]

        returns['dyn_tads'] += [dyn_weight.sum(-1)/np.linalg.norm(dyn_weight, axis=1)]
        returns['dyn_eds'] += [dyn_eig.sum(-1)/np.linalg.norm(dyn_eig, axis=1)]
        returns['dyn_null_eds'] += [dyn_null.sum(-1)/np.linalg.norm(dyn_null, axis=1)]

        returns['dyn_weight_rads'] += [np.linalg.norm(dyn_weight, axis=1)]
        returns['dyn_eig_rads'] += [np.linalg.norm(dyn_eig, axis=1)]
        returns['dyn_null_rads'] += [np.linalg.norm(dyn_null, axis=1)]

        # Compute how many modes are learnable
        # First compute the effective rank based on threshold
        rank_eig, rank_weight = mode_threshold(eig, threshold), mode_threshold(weight_sq, threshold)

        returns['ranks_eig'] += [rank_eig]
        returns['ranks_weight'] += [rank_weight]

    returns = {key: np.array(val).squeeze() for key, val in returns.items()}

    return returns
