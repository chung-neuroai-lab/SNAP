import numpy as np
from scipy import optimize

import torch
from tqdm import tqdm

import jax
import jax.numpy as jnp
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update("jax_enable_x64", True)


@jax.jit
def denom_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    return p*eigs + kappa


@jax.jit
def delta_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return (eigs / denom**2).sum()


@jax.jit
def gamma_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return (p*eigs**2 / denom**2).sum()


@jax.jit
def eff_lambda(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    eff_reg = kappa - kappa * (eigs/denom).sum()
    return eff_reg


# Definition of kappa and its derivatives
@jax.jit
def kappa_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return kappa - reg - kappa * np.sum(eigs/denom)


kappa_prime = jax.jit(jax.grad(kappa_fn, argnums=0))
kappa_pprime = jax.jit(jax.grad(kappa_prime, argnums=0))


def solve_kappa_gamma(pvals, reg, eigs, weights_sq):

    eigs = np.abs(eigs)

    fun, fprime, fprime2 = kappa_fn, kappa_prime, kappa_pprime

    if type(reg) not in [list, np.ndarray]:
        reg = [reg] * len(pvals)
    reg = np.array(reg)

    kappa_vals = np.zeros(len(pvals))
    gamma_vals = np.zeros(len(pvals))
    eff_regs = np.zeros(len(pvals))
    for i, (p, lamb) in enumerate(zip(pvals, reg)):
        args = (p, lamb*p, eigs, weights_sq)

        kappa_0 = lamb + np.sum(eigs)  # When p = 0
        kappa_1 = lamb                 # When p is infty

        kappa_vals[i] = optimize.root_scalar(fun,
                                             fprime=fprime,
                                             fprime2=fprime2,
                                             args=args,
                                             x0=kappa_0,
                                             x1=kappa_1,
                                             method='newton',
                                             xtol=1e-12, maxiter=200).root

        gamma_vals[i] = gamma_fn(kappa_vals[i], *args)
        eff_regs[i] = eff_lambda(kappa_vals[i], *args) / p

    kappa_vals = np.abs(np.nan_to_num(kappa_vals))
    gamma_vals = np.nan_to_num(gamma_vals)
    eff_regs = np.nan_to_num(eff_regs)
    eff_regs = eff_regs + 1e-14

    return np.array(kappa_vals), np.array(gamma_vals), np.array(eff_regs)


def gen_error_theory(eigs, weights, reg, pvals=None, **kwargs):

    # Number of classes
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    C = weights.shape[-1]

    # Sample size for theory
    if pvals is None:
        pvals = np.logspace(np.log10(5), np.log10(len(eigs)-5), 50)

    # Absolute value of eigs improves numerical stability
    eigs = np.abs(eigs)
    weights_sq = (weights**2).sum(-1)

    # Solve for self-consistent equation
    kappa, gamma, eff_regs = solve_kappa_gamma(pvals, reg, eigs, weights_sq)

    # Calculate generalization and training error
    prefactor_gen = kappa ** 2 / (1 - gamma)
    prefactor_tr = eff_regs**2 / kappa**2

    errors = {'pvals_theory': pvals,
              'kappa': kappa,
              'gamma': gamma,
              'eff_regs': eff_regs,
              'mode_err_theory': np.zeros((len(pvals), len(eigs))),
              'gen_theory': np.zeros((len(pvals), C)),
              'tr_theory': np.zeros((len(pvals), C)),
              'r2_theory': np.zeros((len(pvals), C))
              }

    for i, p in enumerate(pvals):
        mode_err = prefactor_gen[i] * (1 / (p*eigs + kappa[i])**2)
        dyn_weights = mode_err[:, None] * weights**2 / weights_sq.sum()

        for j in range(C):
            # Normalize by L2 norm of target
            gen_err = (dyn_weights[:, j]).sum()
            tr_err = prefactor_tr[i] * gen_err
            r2_score = 1 - gen_err

            errors['gen_theory'][i, j] = gen_err
            errors['tr_theory'][i, j] = tr_err
            errors['r2_theory'][i, j] = r2_score
        errors['mode_err_theory'][i] = mode_err

    return errors


@torch.no_grad()
def regression(feat, y, eigs, weights, pvals=None, cent=False, num_points=5, num_trials=3, reg=1e-14, **kwargs):

    P, N = feat.shape
    C = y.shape[-1]

    # To extract the learning curve divide samples into 10
    if pvals is None:
        pvals = np.concatenate([np.logspace(np.log10(5), np.log10(P-2), 5),
                                np.arange(1, 10)*0.1*P]
                               ).astype(int)
        pvals.sort()

    if type(reg) not in [list, np.ndarray]:
        reg = [reg] * len(pvals)
    reg = np.array(reg)

    # Convert to CUDA
    feat = feat.type(torch.cuda.DoubleTensor)
    y = y.type(torch.cuda.DoubleTensor)
    if cent:
        y -= y.mean(0, keepdim=True)
        feat -= feat.mean(0, keepdim=True)

    if P < N:
        K = feat @ feat.T
        feat = 0
        torch.cuda.empty_cache()
    else:
        K = None

    errors = {'pvals': pvals,
              'P': P,
              'N': N,
              'C': C,
              'reg': reg,
              'cent': cent,

              'gen_errs': np.zeros((num_trials, len(pvals), C)),
              'tr_errs': np.zeros((num_trials, len(pvals), C)),
              'test_errs': np.zeros((num_trials, len(pvals), C)),

              'r2_gen': np.zeros((num_trials, len(pvals), C)),
              'r2_tr': np.zeros((num_trials, len(pvals), C)),
              'r2_test': np.zeros((num_trials, len(pvals), C)),

              'pearson_tr': np.zeros((num_trials, len(pvals), C)),
              'pearson_test': np.zeros((num_trials, len(pvals), C)),
              'pearson_gen': np.zeros((num_trials, len(pvals), C)),

              'gen_norm': np.zeros((num_trials, len(pvals), C)),
              'tr_norm': np.zeros((num_trials, len(pvals), C)),
              'test_norm': np.zeros((num_trials, len(pvals), C)),
              }

    for i, (p, lamb) in enumerate(zip(pvals, reg)):
        for j in range(num_trials):

            from sklearn.model_selection import train_test_split
            idx, idx_test = train_test_split(np.arange(0, P, 1), train_size=p)
            assert len(set(idx)) == p
            assert len(set(idx_test)) == P - p

            y_tr = y[idx]
            y_test = y[idx_test]
            try:
                if p >= N:  # Overdetermined lstsq (K_tr = feat.T @ feat)
                    Id = torch.eye(N, device='cuda', dtype=torch.double)
                    feat_tr = feat[idx]
                    y_hat = feat @ torch.linalg.inv(feat_tr.T@feat_tr + p*lamb*Id) @ feat_tr.T @ y_tr
                elif P >= N:   # Underdetermined lstsq (push through identity, K_tr = feat @ feat.T)
                    Id = torch.eye(p, device='cuda', dtype=torch.double)
                    feat_tr = feat[idx]
                    y_hat = feat @ feat_tr.T @ torch.linalg.inv(feat_tr@feat_tr.T + p*lamb*Id) @ y_tr
                else:  # Underdetermined lstsq (push through identity, K_tr = feat @ feat.T)
                    Id = torch.eye(p, device='cuda', dtype=torch.double)
                    y_hat = K[:, idx] @ torch.linalg.inv(K[idx, :][:, idx] + p*lamb*Id) @ y_tr

            except torch.linalg.LinAlgError:
                print('LinAlgErr, Computing regression with pseudo-inverse (slow)')
                Id = torch.eye(p, device='cuda', dtype=torch.double)
                if P >= N:
                    feat_tr = feat[idx]
                    y_hat = feat @ feat_tr.T @ torch.linalg.pinv(feat_tr@feat_tr.T + p*lamb*Id) @ y_tr
                else:
                    y_hat = K[:, idx] @ torch.linalg.pinv(K[idx, :][:, idx] + p*lamb*Id) @ y_tr

            except Exception as e:
                feat, feat_tr, K, y = 0, 0, 0, 0
                torch.cuda.empty_cache()
                raise e

            y_hat_tr = y_hat[idx]
            y_hat_test = y_hat[idx_test]

            tr_cent = y_tr - y_tr.mean(0, keepdim=True)
            test_cent = y_test - y_test.mean(0, keepdim=True)
            gen_cent = y - y.mean(0, keepdim=True)

            # Compute overall (scalar) normalization factors
            tr_norm = (tr_cent**2).mean(0).sum()
            test_norm = (test_cent**2).mean(0).sum()
            gen_norm = (gen_cent**2).mean(0).sum()

            tr_err = ((y_hat_tr - y_tr)**2).mean(0) / tr_norm
            test_err = ((y_hat_test - y_test)**2).mean(0) / test_norm
            gen_err = ((y_hat - y)**2).mean(0) / gen_norm

            r2_tr = 1 - tr_err
            r2_test = 1 - test_err
            r2_gen = 1 - gen_err

            def pearsonr(pred, target):
                yc = target - target.mean(0, keepdim=True)
                yhatc = pred - pred.mean(0, keepdim=True)
                return (yc*yhatc).sum(0)/torch.sqrt((yc**2).sum(0)*(yhatc**2).sum(0))

            pearson_tr = pearsonr(y_hat_tr, y_tr)
            pearson_test = pearsonr(y_hat_test, y_test)
            pearson_gen = pearsonr(y_hat, y)

            errors['gen_norm'][j, i] = gen_norm.cpu().numpy()
            errors['tr_norm'][j, i] = tr_norm.cpu().numpy()
            errors['test_norm'][j, i] = test_norm.cpu().numpy()

            errors['gen_errs'][j, i] = gen_err.cpu().numpy()
            errors['tr_errs'][j, i] = tr_err.cpu().numpy()
            errors['test_errs'][j, i] = test_err.cpu().numpy()

            errors['r2_gen'][j, i] = r2_gen.cpu().numpy()
            errors['r2_tr'][j, i] = r2_tr.cpu().numpy()
            errors['r2_test'][j, i] = r2_test.cpu().numpy()

            errors['pearson_tr'][j, i] = pearson_tr.cpu().numpy()
            errors['pearson_test'][j, i] = pearson_test.cpu().numpy()
            errors['pearson_gen'][j, i] = pearson_gen.cpu().numpy()

    feat, feat_tr, K, y = 0, 0, 0, 0
    torch.cuda.empty_cache()

    return errors


@torch.no_grad()
def regression_metric(activations, labels, spectrum_dict, **kwargs):

    assert type(labels) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"
    assert labels.get('responses') is not None

    reg_responses_uncent = {layer_key: {} for layer_key in activations.keys()}
    reg_responses_cent = {layer_key: {} for layer_key in activations.keys()}
    for layer_key, layer_act in tqdm(activations.items(), total=len(activations), desc='Layer'):
        for label_key, y in labels.items():
            # Uncentered regression
            eigs = spectrum_dict['uncent'][layer_key]['eigs']
            weights = spectrum_dict['uncent'][layer_key]['weights'][label_key]
            theory = gen_error_theory(eigs, weights, **kwargs)
            errors = regression(layer_act, y, eigs, weights, cent=False, **kwargs)
            errors |= theory
            reg_responses_uncent[layer_key][label_key] = errors

            # Centered regression
            eigs = spectrum_dict['cent'][layer_key]['eigs']
            weights = spectrum_dict['cent'][layer_key]['weights'][label_key]
            theory = gen_error_theory(eigs, weights, **kwargs)
            errors = regression(layer_act, y, eigs, weights, cent=True, **kwargs)
            errors |= theory
            reg_responses_cent[layer_key][label_key] = errors

    return {'uncent': reg_responses_uncent,
            'cent': reg_responses_cent}
