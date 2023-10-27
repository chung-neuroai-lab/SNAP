import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

import numpy as np
from collections import OrderedDict


def hook_activation(name, activation, store_batches=False, pooling=None):
    # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    # If store_batches True, accumulate outputs across batches.

    if pooling is not None:
        pooling, output_dim = pooling.split('_')
        output_dim = eval(output_dim)
    assert pooling in ['MaxPool', 'AvgPool', None]

    def hook(model, input, output):

        if pooling is None:
            def pool_fn(out):
                return out.reshape(len(out), -1)

        elif output.ndim == 4:

            if pooling == 'MaxPool':
                pool_fn = pool_MaxPool(output_dim)

            elif pooling == 'AvgPool':
                pool_fn = pool_AvgPool(output_dim)

        else:
            raise Exception(f'For Max/Avg Pooling, inputs must be convolutinal! Inputs have dim {output.ndim}')

        output = pool_fn(output.cuda().type(torch.cuda.DoubleTensor)).detach().cpu()
        assert output.ndim == 2, "Activations must be 1d vectors"
        # print(model._get_name(), output.shape)

        if store_batches:
            activation[name] += [output]
        else:
            activation[name] = output

    return hook


def get_layer(model, layer_name):

    module = model
    for part in layer_name.split('.'):
        module = module._modules.get(part)

    return module


def register_hooks(model, layers, activation, store_batches=False, pooling=None):

    hooks = {}
    for layer in layers:
        module = get_layer(model, layer)
        hooks[layer] = module.register_forward_hook(hook_activation(layer, activation, store_batches, pooling))

    return hooks


class TorchWrapper:

    def __init__(self, model, layers: list[str], identifier=None, activation_pooling=None):

        self.identifier = identifier or model.__class__.__name__
        self.activation_pooling = activation_pooling

        self.model = model
        self.model.eval()

        for p in model.parameters():
            p.requires_grad = False

        self.layers = layers
        if self.layers is not None:
            self.activations = OrderedDict()
            for layer in layers:
                self.activations[layer] = []

    @torch.no_grad()
    def __call__(self, input):

        # Register hooks
        if self.layers is not None:
            hooks = self._register_hooks(store_batches=False)

        # Evaluate model
        out = self.model(input)

        # Remove hooks
        if self.layers is not None:
            for layer, hook in hooks.items():
                hook.remove()

        torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def eval(self, data_loader):

        loss_fn = torch.nn.CrossEntropyLoss()

        pred_acc, pred_loss, total_num = 0., 0., 0.
        for x, y in data_loader:

            assert len(y.shape) == 2, "labels are assumed to be one-hot encoded"
            y = torch.argmax(y, dim=1)

            with autocast():
                out = self.model(x)
                pred_acc += out.argmax(1).eq(y).sum().cpu().item()
                pred_loss += loss_fn(out, y).cpu().item()
                total_num += x.shape[0]

        pred_acc = pred_acc / total_num * 100
        pred_loss = pred_loss / total_num

        model_acc = dict(pred_acc=pred_acc,
                         pred_loss=pred_loss
                         )

        return model_acc

    def _register_hooks(self, store_batches=False) -> dict[str, None]:

        return register_hooks(self.model, self.layers, self.activations, store_batches, self.activation_pooling)

    def get_layer(self, layer):

        return get_layer(self.model, layer)


def pool_MaxPool(output_dim: tuple[int, int]):

    assert isinstance(output_dim, tuple) and isinstance(output_dim[0], int) and isinstance(output_dim[1], int)

    @torch.no_grad()
    def pool_fn(out):

        B, C, H, W = out.shape  # Channels first (B, C, H, W)
        if H*W*C > np.prod(output_dim)*C:
            out = F.adaptive_max_pool2d(out, output_dim)
            assert len(out) == B

        return out.reshape(B, -1)

    return pool_fn


def pool_AvgPool(output_dim: tuple[int, int]):

    assert isinstance(output_dim, tuple) and isinstance(output_dim[0], int) and isinstance(output_dim[1], int)

    @torch.no_grad()
    def pool_fn(out):

        B, C, H, W = out.shape  # Channels first (B, C, H, W)
        if H*W*C > np.prod(output_dim)*C:
            out = F.adaptive_avg_pool2d(out, output_dim)
            assert len(out) == B

        return out.reshape(B, -1)

    return pool_fn


def pool_PCA(output_dim):

    assert isinstance(output_dim, int)

    @torch.no_grad()
    def pool_fn(out):

        B, C, H, W = out.shape  # Channels first (B, C, H, W)
        if C*H*W < output_dim:
            return out.reshape(B, -1)

        out = out.type(torch.cuda.DoubleTensor).reshape(B, -1)
        _, _, V = torch.linalg.svd(out, full_matrices=False)
        out = torch.matmul(out, V[:output_dim].T)

        return out

    return pool_fn


def pool_RandProj(output_dim: int):

    assert isinstance(output_dim, int)

    @torch.no_grad()
    def pool_fn(out):

        B, C, H, W = out.shape  # Channels first (B, C, H, W)
        flat_out_dim = C*H*W

        out = out.reshape(B, -1)

        if flat_out_dim < output_dim:
            return out

        is_oom = True
        N = 1

        projected_out = []
        while is_oom:

            try:
                for i in range(N):
                    rand_proj = torch.randn(size=(flat_out_dim, output_dim//N), device=out.device).type(out.dtype)
                    rand_proj = rand_proj / torch.linalg.norm(rand_proj, axis=1, keepdim=True)

                    projected_out += [out @ rand_proj]

                    is_oom = False

            except torch.cuda.OutOfMemoryError:
                N = 2*N
                print(f"Cuda OOM - Trying {N} batched random projection")

        projected_out = torch.cat(projected_out, dim=1)
        assert len(projected_out) == B

        return projected_out

    return pool_fn
