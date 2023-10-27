import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

import os
import pickle


def get_neural_data(region, dataset='dicarlo', image_type='original'):

    assert region in ['V1', 'V2', 'V4', 'IT']

    if dataset == 'dicarlo':

        identifier = 'freemanziemba2013' if region in ['V1', 'V2'] else 'majajhong2015'
        df = get_brainscore(identifier=identifier, region=region)

    elif dataset == 'cadena' and region == 'V1':

        # Data from https://github.com/sacadena/Cadena2019PlosCB
        data_path = './Cadena_PlosCB19_data/data_binned_responses/cadena_ploscb_data.pkl'

        if not os.path.exists(data_path):

            url = 'https://doid.gin.g-node.org/2e31e304e03d6357c98ac735a1fe5788/2e31e304e03d6357c98ac735a1fe5788.zip'
            out_dir = "./Cadena_PlosCB19_data"
            os.makedirs(out_dir, exist_ok=True)

            import wget
            import zipfile
            filename = wget.download(url, out=out_dir)
            with zipfile.ZipFile(filename, "r") as zip:
                zip.extractall(out_dir)

        df = get_cadena(data_path, image_type)

    else:
        raise Exception(f'No such dataset with dataset {dataset} and region {region}!')

    # Load images vs responses data_loader
    loader_kwargs = {'batch_size': 800,
                     'shuffle': False,
                     'num_workers': 4,
                     'pin_memory': True,
                     'onehot': True,
                     'labels_from': 'neural_activity'
                     }

    # Get the DataLoader for Neural Responses and the entire ordered data
    data_loader_neural, label_map = get_dataloader(df, **loader_kwargs)
    data_loader_neural = LoaderTORCH(data_loader_neural, cuda=True)
    images, responses = get_ordered_data(data_loader_neural())

    print(images.shape, responses.shape)

    labels = {'responses': responses
              }

    return data_loader_neural, images, labels


def get_dataloader(df,
                   labels_from='categories',
                   onehot=False,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225),
                   **loader_kwargs):

    image_files = df.image_files
    image_files = np.array(image_files.to_list())

    if labels_from == 'categories':
        image_categories = df.image_categories
        label_names = image_categories
        task = 'categorical'
    elif labels_from == 'names':
        image_names = df.image_names.values
        label_names = image_names
        task = 'categorical'
    elif labels_from == 'neural_activity':
        label_names = ""
        task = 'regression'
    else:
        raise Exception

    target_transform = None
    if task == 'categorical':
        label_map = {lb: label for lb, label in zip(set(label_names), range(len(label_names)))}
        labels = np.vectorize(label_map.get)(label_names)
        if onehot:
            def target_transform(y): return torch.eye(len(label_map))[y]
    else:
        # task == 'regression':
        label_map = None
        labels = np.stack(df.mean_responses).astype(np.float32)

    transform = [transforms.Resize((224, 224), antialias=True),
                 transforms.ConvertImageDtype(torch.float32),
                 ]
    if mean is not None and std is not None:
        transform += [transforms.Normalize(mean=mean, std=std)]
    transform = transforms.Compose(transform)

    ds = BrainscoreImageDataset(image_files, labels,
                                transform=transform,
                                target_transform=target_transform)

    data_loader = DataLoader(ds, **loader_kwargs)

    return data_loader, label_map


class BrainscoreImageDataset(Dataset):
    def __init__(self, images, img_labels, transform=None, target_transform=None):
        self.images = images
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

        if len(self.images.shape) == 4:
            # Cadena dataset images are tensors
            self.images = preprocess_cadena(self.images)
            # print('Custom prepreprocessing is used!!!')
            # self.images = preprocess_cadena_3(self.images)

        else:
            # DiCarlo dataset images are file paths
            self.images = np.array([str(file) for file in self.images])
            self.images = [read_image(str(file), ImageReadMode(3)) for file in self.images]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class LoaderTORCH:
    def __init__(self, loader, cuda=False):
        self.loader = loader
        self.cuda = cuda

    def __call__(self):
        return self.torch_loader(self.loader, self.cuda)

    def torch_loader(self, loader, cuda=False):
        for data in tqdm(loader, total=len(loader), desc="Batch"):

            if cuda:
                yield data[0].cuda(), data[1].cuda()
            else:
                yield data[0], data[1]


def get_ordered_data(data_loader):

    X, Y = [], []
    for x, y in data_loader:
        x, y = x.cpu(), y.cpu()
        X += [x]
        Y += [y]

    images = torch.cat(X)
    labels = torch.cat(Y)

    return images, labels


def get_brainscore(identifier="majajhong2015", region='IT'):

    if identifier == "majajhong2015":
        import brainscore.benchmarks.majajhong2015 as majajhong2015
        assert region in ['IT', 'V4']

        majajhong2015 = majajhong2015.load_assembly(average_repetitions=False, region=region, access='public')
        neural_data = majajhong2015
        visual_degree = 8

    elif identifier == "freemanziemba2013":
        import brainscore.benchmarks.freemanziemba2013 as freemanziemba2013
        assert region in ['V1', 'V2']

        freemanziemba2013 = freemanziemba2013.load_assembly(average_repetitions=False, region=region, access='public')
        neural_data = freemanziemba2013
        visual_degree = 4

    else:
        raise Exception(f"Only {['majajhong2015', 'freemanziemba2013']} allowed.")

    # Get individual image ids
    image_ids = list(set(neural_data.image_id.data))

    # Create dictionary for repetitions, responses, image files, names and categories for each distinct image
    repetitions = {}
    neural_responses = {}
    image_files = {}
    image_names = {}
    image_categories = {}

    from brainscore.benchmarks.screen import place_on_screen
    stimulus_set = place_on_screen(neural_data.stimulus_set,
                                   target_visual_degrees=8,
                                   source_visual_degrees=visual_degree)

    for image_id in tqdm(image_ids, desc="Getting Neural Data"):

        data_image = neural_data.sel(image_id=image_id)

        repetition = len(data_image.presentation.repetition)
        neural_response = data_image.values.squeeze()
        image_file = stimulus_set.stimulus_paths[image_id]

        if identifier == "majajhong2015":
            image_name = data_image.object_name.values
            image_category = data_image.category_name.values
        elif identifier == "freemanziemba2013":
            image_name = data_image.texture_family.values
            image_category = data_image.texture_type.values
        else:
            raise Exception(f"Only {['majajhong2015', 'freemanziemba2013']} allowed.")

        repetitions[image_id] = repetition
        neural_responses[image_id] = neural_response
        image_files[image_id] = image_file
        image_names[image_id] = image_name[0]
        image_categories[image_id] = image_category[0]

    # Create pandas dataframe
    data = {'image_ids': image_ids,
            'repetition': repetitions.values(),
            'neural_responses': neural_responses.values(),
            'image_names': image_names.values(),
            'image_categories': image_categories.values(),
            'image_files': image_files.values()}

    # Sort according to categories
    df = pd.DataFrame(data=data)
    df = df.sort_values(by=['image_categories', 'image_files'], ascending=[True, True], ignore_index=True)

    # Compute mean_responses and extract features
    df.insert(2, "mean_responses", df.neural_responses.map(lambda x: x.mean(0)))
    df.insert(3, "std_responses", df.neural_responses.map(lambda x: x.std(0)))

    return df


def get_cadena(data_path, image_type='original'):

    if len(image_type.split('_')) == 2:
        image_type, subsample = image_type.split('_')
        subsample = int(subsample)
    else:
        subsample = None

    all_image_files, all_responses = process_cadena_data(data_path)

    # Select the stimuli type
    if image_type == 'all':
        image_files = np.concatenate([img for img in all_image_files.values()], axis=0)
        responses = np.concatenate([img for img in all_responses.values()], axis=1)
    else:
        image_files = all_image_files[image_type]
        responses = all_responses[image_type]

    image_files = image_files[:, None, :, :]
    image_files = np.tile(image_files, [1, 3, 1, 1])
    neural_responses = responses.swapaxes(0, 1)
    assert np.isnan(neural_responses).sum() == 0

    if subsample is not None:

        np.random.seed(65)
        idx = np.random.choice(np.arange(0, len(image_files), 1), size=subsample, replace=False)

        neural_responses = neural_responses[idx]
        image_files = image_files[idx]

    data = {'neural_responses': list(neural_responses),
            'image_files': image_files.tolist()}

    # Sort according to categories
    df = pd.DataFrame(data=data)

    # Compute mean_responses and extract features
    df.insert(2, "mean_responses", df.neural_responses.map(lambda x: x.mean(0)))
    df.insert(3, "std_responses", df.neural_responses.map(lambda x: x.std(0)))

    return df


def process_cadena_data(data_path):
    """
    This code is taken from: https://github.com/nathankong/robustness_primary_visual_cortex
    """

    CADENA_RAW_DATA = data_path
    cadena_data = pickle.load(open(CADENA_RAW_DATA, "rb"))

    def split_into_two_trials_cadena(responses):
        """
        Since some neurons only have two trials, we find the images that have at least
        two trials for all the neurons (the trials where the neuron does not have a
        response is coded as `NaN' in the data set. We then organize the responses
        into a numpy array of dimensions (2 trials, n_stim, n_neurons).

        Input:
            responses     : numpy array of dimensions (4 trials, n_stim, n_neurons)
            valid_img_idx : numpy array of booleans indicating which stimuli were
                            included and which were removed (removed if there are less
                            than two trials for the image)
        """
        valid_img_idx = np.ones((responses.shape[1],)).astype(bool)
        responses_by_trial = []
        for i in range(responses.shape[1]):
            by_trial = []
            for j in range(responses.shape[0]):
                if np.sum(np.isnan(responses[j, i, :])) == 0:
                    by_trial.append(responses[j, i, :])
            # If the number of trials for stim is less than 2, remove the stim
            if (len(by_trial) < 2):
                # print(i)
                valid_img_idx[i] = 0
            # Otherwise, average each half of all the trials and create two
            # `pseudo-trials' for the stim
            else:
                by_trial = np.array(by_trial)
                half_trials = int(len(by_trial)/2)
                r1 = np.mean(by_trial[:half_trials, :], axis=0)[np.newaxis, :]
                r2 = np.mean(by_trial[half_trials:, :], axis=0)[np.newaxis, :]
                responses_by_trial.append(np.vstack((r1, r2)))

        responses_by_trial = np.transpose(np.array(responses_by_trial), (1, 0, 2))
        return responses_by_trial, valid_img_idx

    all_image_files = {}
    all_responses = {}
    for img_type in ["original", "conv1", "conv2", "conv3", "conv4"]:

        img_idx = np.array(cadena_data["image_types"] == img_type).flatten()
        responses = cadena_data["responses"][:, img_idx, :]

        responses, valid_img_idx = split_into_two_trials_cadena(responses)
        image_files = cadena_data["images"][img_idx, :, :][valid_img_idx]

        all_image_files[img_type] = image_files
        all_responses[img_type] = responses

    return all_image_files, all_responses


def preprocess_cadena_3(images):

    images = torch.as_tensor(images, dtype=torch.uint8)
    images = transforms.functional.pad(images, padding=(42, 42), padding_mode='edge')

    return images


def preprocess_cadena_2(images):

    images = torch.as_tensor(images, dtype=torch.uint8)
    images = transforms.functional.pad(images, padding=(42, 42), fill=0, padding_mode='constant')

    return images


def preprocess_cadena(images):
    """
    This code is taken from: https://github.com/nathankong/robustness_primary_visual_cortex
    """

    """
    Cropping to images to 80 px (as done in Cadena et al. 2019), which is about
    1.1 degrees in monkey "land".  Then resize the stimuli to 40 px (assuming
    that 224 px in deep net "land" is 6.4 degrees. So 40 px is about 1.1 degrees.

    Output:
        dataloader : torch.utils.data.DataLoader for the stimuli
    """

    input_height = 224
    input_width = 224
    pad_left = int(np.floor((input_width-40)/2.))
    pad_top = int(np.floor((input_height-40)/2.))
    pad_right = int(np.ceil((input_width-40)/2.))
    pad_bottom = int(np.ceil((input_height-40)/2.))
    pad_amount = (pad_left, pad_top, pad_right, pad_bottom)

    images = torch.as_tensor(images, dtype=torch.float32)
    images = transforms.functional.center_crop(images, 80)
    images = transforms.functional.resize(images, 40, antialias=True)
    images = transforms.functional.pad(images, padding=pad_amount, fill=0, padding_mode='constant')
    images = transforms.functional.normalize(images, mean=[0, 0, 0], std=[1, 1, 1])

    return images
