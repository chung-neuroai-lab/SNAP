from typing import Callable, List

import torch
import numpy as np

from .metrics import compute_spectrum


class Experiment:

    def __init__(self, model, metric_fns: List[Callable] = [], rand_proj_dim=None):
        self.model = model
        self.metric_fns = metric_fns
        self.rand_proj_dim = rand_proj_dim

    @torch.no_grad()
    def get_activations(self, dataloader) -> dict[str, torch.Tensor]:

        print("Getting layer activations...")

        self.activations = {}

        for x, _ in dataloader:

            _ = self.model(x.cuda())
            act = self.model.activations

            if len(self.activations) == 0:
                self.activations = {key: [val.cpu()] for key, val in act.items()}
            else:
                self.activations = {key: self.activations[key] + [val.cpu()] for key, val in act.items()}

        for key, val in self.activations.items():
            self.activations[key] = torch.cat(val)

        if self.rand_proj_dim is not None:
            print("Random projection...")

            rand_proj = RandProj(self.rand_proj_dim)

            for key, act in self.activations.items():
                act = act.cuda().type(torch.cuda.DoubleTensor)
                act = rand_proj(act).cpu()
                self.activations[key] = act

        return self.activations

    @torch.no_grad()
    def compute_metrics(self, images, labels, activations=None, max_points=10000, **kwargs):

        assert type(labels) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"
        print("Computing metrics...")

        if activations is not None:
            self.activations = activations

        # Flatten images
        P = len(images)
        images = images.reshape(P, -1)

        # Allow maximum {max_points} points
        if max_points < P:
            # Randomly sample {max_points} points
            idx = np.random.choice(np.arange(0, P, 1), size=max_points, replace=False)
            assert len(set(idx)) == max_points  # Check that all points are unique

            # Restrict the number of data points
            images = images[idx]
            labels = {key: lb[idx] for key, lb in labels.items()}
            for layer_key, val in self.activations.items():
                self.activations[layer_key] = val[idx]

        # Include images and responses as activations as well!
        self.activations['image_layer'] = images
        if labels.get('responses') is not None:
            self.activations['response_layer'] = labels['responses']

        # Create the data dictionary to store all metric outputs
        data_dict = {'layers': list(self.activations.keys())}
        print(f"Computing metrics for {data_dict['layers']}")
        print({layer_key: val.shape for layer_key, val in self.activations.items()})

        # First compute spectrum, weights, kt alignment etc.
        print("Computing spectrum...")
        spectrum_dict = compute_spectrum(self.activations, images, labels, **kwargs)
        data_dict['spectrum'] = spectrum_dict

        # Compute all the metrics in self.metric_fns
        import copy
        metric_kwargs = {'spectrum_dict': copy.deepcopy(spectrum_dict),
                         'images': images,
                         'labels': labels,
                         } | kwargs

        for metric_fn in self.metric_fns:
            print(f"Computing {metric_fn.__name__}...")
            data_dict[metric_fn.__name__] = metric_fn(activations=self.activations, **metric_kwargs)

        print("Metric Computation completed!")

        return data_dict


def RandProj(output_dim: int):

    assert isinstance(output_dim, int)

    @torch.no_grad()
    def pool_fn(inp):

        assert len(inp.shape) == 2, "Inputs to random projection must be 2-dimensional!"

        P, input_dim = inp.shape

        if input_dim < output_dim:
            return inp

        is_oom = True
        N = 1

        projected_out = []
        while is_oom:

            try:
                for i in range(N):
                    rand_proj = torch.randn(size=(input_dim, output_dim//N), device=inp.device).type(inp.dtype)
                    rand_proj = rand_proj / torch.linalg.norm(rand_proj, axis=1, keepdim=True)

                    projected_out += [inp @ rand_proj]

                    is_oom = False

            except torch.cuda.OutOfMemoryError:
                N = 2*N
                print(f"Cuda OOM - Trying {N} batched random projection")

        projected_out = torch.cat(projected_out, dim=1)
        assert len(projected_out) == P

        return projected_out

    return pool_fn
