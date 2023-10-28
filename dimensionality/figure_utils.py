import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib as mpl

DATA_ROOT = './data/'


def get_processed_data_figs(region_list, pooling_list, trained=True, exclude_models=None):
    """
    Loads in the processed data and adds a few additional keys to all_reg_hist
    """

    pretrained = {True: 'pretrained',
                  False: 'untrained'}

    all_reg_hist = {key: {key_pool: {} for key_pool in pooling_list} for key in region_list}
    all_processed_data = {key: {key_pool: {} for key_pool in pooling_list} for key in region_list}
    all_data_kwargs = {key: {key_pool: {} for key_pool in pooling_list} for key in region_list}
    for region in region_list:
        for pooling in pooling_list:
            processed_data_name = DATA_ROOT+'processed/'
            processed_data_name += f'{region}_{pooling}_{pretrained[trained]}.npz'

            data = np.load(processed_data_name, allow_pickle=True)
            all_reg_hist[region][pooling] = data['all_reg_hist'].tolist()

            all_processed_data[region][pooling] = data['all_processed_data'].tolist()
            all_data_kwargs[region][pooling] = data['all_data_kwargs'].tolist()

            # Add model_layer values to all_reg_hist
            model_layer_combined = ['%s_%s' % (m, l) for (m, l) in zip(all_reg_hist[region][pooling]['models'],
                                                                       all_reg_hist[region][pooling]['layers'])]
            all_reg_hist[region][pooling]['models_plus_layer'] = model_layer_combined
            layer_norm_idx_dict = {}
            for model_name, model_info in all_processed_data[region][pooling].items():
                num_layers = len(model_info['layers'])
                layer_norm_idx_dict_temp = {'%s_%s' % (model_name, l): l_idx/num_layers for
                                            l_idx, l in enumerate(model_info['layers'])}
                layer_norm_idx_dict.update(layer_norm_idx_dict_temp)

            all_reg_hist[region][pooling]['layer_depth_normalized'] \
                = [layer_norm_idx_dict[ml] for ml in all_reg_hist[region][pooling]['models_plus_layer']]

            # Add generalization error for all p values to all_reg_hist
            gen_err_dict = {}
            for model_name, model_info in all_processed_data[region][pooling].items():
                gen_err_dict_tmp = {'%s_%s' % (model_name, l): model_info['gen_errs'][l_idx, :] for
                                    l_idx, l in enumerate(model_info['layers'])}
                gen_err_dict.update(gen_err_dict_tmp)
            all_reg_hist[region][pooling]['gen_errs'] = np.array([gen_err_dict[ml] for ml in
                                                                  all_reg_hist[region][pooling]['models_plus_layer']])

            # Filter out all of the excluded models from all_reg_hist.
            if exclude_models is not None:
                for model_name in exclude_models:
                    include_m_idx = all_reg_hist[region][pooling]['models'] != model_name
                    for key in all_reg_hist[region][pooling].keys():
                        all_reg_hist[region][pooling][key] = np.array(all_reg_hist[region][pooling][key])[include_m_idx]

    return all_reg_hist, all_processed_data, all_data_kwargs


def plot_region_contours(all_reg_hist,
                         region_list,
                         pooling,
                         p_idx=-2,
                         x_lims=None,
                         y_lims=None,
                         coloring='final_scores',
                         c_bar_label=r'$E_g(p)$',
                         c_map_min=None,
                         c_map_max=None,
                         c_map_min_contours=None,
                         c_map_max_contours=None,
                         cmap=plt.cm.plasma,
                         cmap_contours=None,
                         model_subset=None,
                         marker_size=2,
                         marker_style='.',
                         ax_handle=None,
                         save_figures=False,
                         make_contours=True,
                         figure_root='./data/figures/'):
    """
    Makes a figure of sqrt(D) vs. R showing contour lines for the
    theoretical generalization error.
    """

    if ax_handle is None:
        plt.figure(figsize=(4*int(len(region_list)/2), 3*int(len(region_list)/2)))

    region_data_plotted = {}
    for region_idx, region in enumerate(region_list):
        if ax_handle is None:
            ax = plt.subplot(int(len(region_list)/2), int(len(region_list)/2), region_idx+1)
        else:
            ax = ax_handle[region_idx]

        all_models = copy.deepcopy(all_reg_hist[region][pooling])

        if model_subset is not None:
            model_idx = [m_idx for m_idx, m in enumerate(all_models['models']) if m in model_subset]
            for key in all_models.keys():
                all_models[key] = np.array([all_models[key][m_idx] for m_idx in model_idx])

        # Define the colormap and normalization for the datapoints
        cmap_data = all_models[coloring]
	if isinstance(cmap_data, list):
            cmap_data = np.array(cmap_data)
        if len(cmap_data.shape) == 2:
            cmap_data = cmap_data[:, p_idx]
        if c_map_max is None:
            c_map_max_region = np.max(cmap_data)
        else:
            c_map_max_region = c_map_max
        if c_map_min is None:
            c_map_min_region = np.min(cmap_data)
        else:
            c_map_min_region = c_map_min

        norm_data = mpl.colors.Normalize(vmax=c_map_max_region,
                                         vmin=c_map_min_region)

        # Colormap and normalization for the contours
        if c_map_min_contours is None:
            c_map_min_contours_region = c_map_min_region
        else:
            c_map_min_contours_region = c_map_min_contours
        if c_map_max_contours is None:
            c_map_max_contours_region = c_map_max_region
        else:
            c_map_max_contours_region = c_map_max_contours
        if cmap_contours is None:
            cmap_contours = cmap
        norm_contours = mpl.colors.Normalize(vmax=c_map_max_contours_region,
                                             vmin=c_map_min_contours_region)

        # Extract the data for the scatter plots
        r_wtilda = all_models['dyn_weight_rads']
        sqrtd_wtilda = all_models['dyn_tads']

        # Just a sanity check that we are plotting the right thing
        pval = all_models['pvals_mode'][:, p_idx]
        assert len(set(pval)) == 1, 'Detected Multiple Sample Values!'

        # Create the first scatter plot
        ax_sc = ax.scatter(r_wtilda[:, p_idx], sqrtd_wtilda[:, p_idx],
                           c=cmap_data,
                           cmap=cmap, norm=norm_data,
                           marker=marker_style, s=marker_size,
                           )

        if x_lims is None:
            current_xlim = list(ax.get_xlim())  # plt.xlim()
            if np.min(r_wtilda[:, p_idx]) < current_xlim[0]:
                current_xlim[0] = np.min(r_wtilda[:, p_idx]) - 0.0001
            if np.max(r_wtilda[:, p_idx]) > current_xlim[1]:
                current_xlim[1] = np.max(r_wtilda[:, p_idx]) + 0.0001
        else:
            current_xlim = x_lims[region]

        if y_lims is None:
            current_ylim = list(ax.get_ylim())  # plt.ylim()
            if np.min(sqrtd_wtilda[:, p_idx]) < current_ylim[0]:
                current_ylim[0] = np.min(sqrtd_wtilda[:, p_idx]) - 0.0001
            if np.max(sqrtd_wtilda[:, p_idx]) > current_ylim[1]:
                current_ylim[1] = np.max(sqrtd_wtilda[:, p_idx]) + 0.0001
        else:
            current_ylim = y_lims[region]

        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)

        if make_contours:
            # Create a grid of x and y values
            x = np.linspace(current_xlim[0], current_xlim[1], 200)
            y = np.linspace(current_ylim[0], current_ylim[1], 200)
            X, Y = np.meshgrid(x, y)

            # Compute z values from x and y
            Z = X * Y

            # Create a contour plot
            contour = ax.contour(X, Y, Z, cmap=cmap_contours,
                                 norm=norm_contours,
                                 vmin=c_map_min_contours_region,
                                 vmax=c_map_max_contours_region)

            # Label the contour lines
            plt.clabel(contour, inline=True, fontsize=10)

        ax.set_title('%s' % region)
        ax.set_xlabel('$R_{em}$')
        ax.set_ylabel(r'$\sqrt{D_{em}}$')

        # Add a colorbar
        plt.colorbar(ax_sc,
                     label=c_bar_label)

        region_data_plotted[region] = {'all_models': all_models,
                                       'ax': ax,
                                       'p_idx': p_idx,
                                       'cmap_data': cmap_data}

    plt.tight_layout()

    if (model_subset is None) and save_figures:
        plt.savefig(figure_root + 'AllRegions_Trained_RvsDContour_Color%s.pdf' % coloring)
        plt.savefig(figure_root + 'AllRegions_Trained_RvsDContour_Color%s.png' % coloring)
    elif save_figures:
        plt.savefig(
            figure_root + 'AllRegions_Trained_RvsDContour_Color%s_Models%s.pdf' %
            (coloring, '|'.join(model_subset)))
        plt.savefig(
            figure_root + 'AllRegions_Trained_RvsDContour_Color%s_Models%s.png' %
            (coloring, '|'.join(model_subset)))

    return region_data_plotted
