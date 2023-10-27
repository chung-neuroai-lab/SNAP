import os
import torch
from torchvision.models import (alexnet, AlexNet_Weights,
                                vgg11, VGG11_Weights,
                                vgg16, VGG16_Weights,
                                vgg19, VGG19_Weights,
                                resnet18, ResNet18_Weights,
                                resnet50, ResNet50_Weights,
                                resnet101, ResNet101_Weights,
                                resnet152, ResNet152_Weights,
                                convnext_small, ConvNeXt_Small_Weights,
                                convnext_base, ConvNeXt_Base_Weights,
                                convnext_large, ConvNeXt_Large_Weights,
                                vit_b_16, ViT_B_16_Weights,
                                vit_b_32, ViT_B_32_Weights,
                                vit_l_16, ViT_L_16_Weights,
                                vit_l_32, ViT_L_32_Weights,
                                vit_h_14, ViT_H_14_Weights)
import torch.utils.model_zoo

alexnet_pt_layers = ['features.1',
                     'features.4',
                     'features.7',
                     'features.9',
                     'features.11']

vgg11_pt_layers = ["features.4",
                   "features.7", "features.9",
                   "features.12", "features.14",
                   "features.17", "features.19"]


vgg16_pt_layers = ["features.6", "features.8",
                   "features.11", "features.13", "features.15",
                   "features.18", "features.20", "features.22",
                   "features.25", "features.27", "features.29"]

vgg19_pt_layers = ["features.6", "features.8",
                   "features.11", "features.13", "features.15", "features.17",
                   "features.20", "features.22", "features.24", "features.26",
                   "features.29", "features.31", "features.33", "features.35",]

resnet18_pt_layers = [f'layer1.{i}.relu' for i in range(2)] + \
                     [f'layer2.{i}.relu' for i in range(2)] + \
                     [f'layer3.{i}.relu' for i in range(2)] + \
                     [f'layer4.{i}.relu' for i in range(2)]

resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]

resnet101_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                      [f'layer2.{i}.relu' for i in range(4)] + \
                      [f'layer3.{i}.relu' for i in range(23)] + \
                      [f'layer4.{i}.relu' for i in range(3)]

resnet152_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                      [f'layer2.{i}.relu' for i in range(8)] + \
                      [f'layer3.{i}.relu' for i in range(36)] + \
                      [f'layer4.{i}.relu' for i in range(3)]

convnext_base_pt_layers = ['features.0'] + \
                          [f'features.1.{i}' for i in range(3)] + \
                          ['features.2'] + \
                          [f'features.3.{i}' for i in range(3)] + \
                          ['features.4'] + \
                          [f'features.5.{i}' for i in range(3)] + \
                          ['features.6'] + \
                          [f'features.7.{i}' for i in range(3)]

vit_base_pt_layers = [f'encoder.layers.encoder_layer_{i}' for i in range(12)]
vit_large_pt_layers = [f'encoder.layers.encoder_layer_{i}' for i in range(24)]
vit_huge_pt_layers = [f'encoder.layers.encoder_layer_{i}' for i in range(32)]


def get_model(name, pretrained=False, device='cuda', **kwargs):

    if name == 'alexnet':
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = alexnet(weights=weights)
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        layers = alexnet_pt_layers

    elif name == 'vgg11':
        weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg11(weights=weights)
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        layers = vgg11_pt_layers

    elif name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16(weights=weights)
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        layers = vgg16_pt_layers

    elif name == 'vgg19':
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg19(weights=weights)
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        layers = vgg19_pt_layers

    elif name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        layers = resnet18_pt_layers

    elif name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = resnet50(weights=weights)
        layers = resnet50_pt_layers

    elif name == 'resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = resnet101(weights=weights)
        layers = resnet101_pt_layers

    elif name == 'resnet152':
        weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = resnet152(weights=weights)
        layers = resnet152_pt_layers

    elif name == 'convnext_small':
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = convnext_small(weights=weights)
        layers = convnext_base_pt_layers

    elif name == 'convnext_base':
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = convnext_base(weights=weights)
        layers = convnext_base_pt_layers

    elif name == 'convnext_large':
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = convnext_large(weights=weights)
        layers = convnext_base_pt_layers

    elif name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = vit_b_16(weights=weights)
        layers = vit_base_pt_layers

    elif name == 'vit_b_32':
        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = vit_b_32(weights=weights)
        layers = vit_base_pt_layers

    elif name == 'vit_l_16':
        weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = vit_l_16(weights=weights)
        layers = vit_large_pt_layers

    elif name == 'vit_l_32':
        weights = ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = vit_l_32(weights=weights)
        layers = vit_large_pt_layers

    elif name == 'vit_h_14':
        weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model = vit_h_14(weights=weights)
        layers = vit_huge_pt_layers

    elif name.split('_')[0] == 'cornet':
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model_letter = name.split("_")[1]
        assert model_letter in ['r', 'z', 'rt', 's']
        model, layers = get_cornet(model_letter=model_letter, pretrained=pretrained, device=device)

    elif name.split('_')[0] == 'vonenet':
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model_arch = name.split("_")[1]
        assert model_arch in ['resnet50', 'cornets', 'alexnet']
        model, layers = get_vonenet(model_arch=model_arch, pretrained=pretrained, device=device)

    elif 'efficientnet' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['efficientnet_b0', 'efficientnet_b4']
        model, layers = get_efficientnet(name, pretrained=pretrained)

    elif name == 'mobilenet_v2':
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model, layers = get_mobilenet(pretrained=pretrained)

    elif 'densenet' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']
        model, layers = get_densenet(name, pretrained=pretrained)

    elif 'wide_resnet' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['wide_resnet50_2', 'wide_resnet101_2']
        model, layers = get_wideresnet(name, pretrained=pretrained)

    elif name == 'barlowtwins':
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        model, layers = get_barlowtwins(pretrained=pretrained, device=device)

    elif 'simclr' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['simclr_resnet50w4', 'simclr_resnet50w1', 'simclr_resnet101']
        model, layers = get_SimCLR(name, pretrained=pretrained, device=device)

    elif 'robust' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['robust_resnet50_l2_3', 'robust_resnet50_linf_4', 'robust_resnet50_linf_8']
        model, layers = get_robustness(name, pretrained=pretrained, device=device)

    elif 'moco' in name:
        identifier = name + ('|imagenet_trained' if pretrained else '|untrained')
        assert name in ['moco_resnet50']
        model, layers = get_moco_resnet50(pretrained=pretrained)

    else:
        raise Exception(f'Invalid Model Selection: {name}')

    return model.to(device), layers, identifier


def get_cornet(model_letter, pretrained=False, device='cuda'):

    # model_letters: z r rt s
    cornet_pt_layers = ['module.V1.output',
                        'module.V2.output',
                        'module.V4.output',
                        'module.IT.output']

    try:
        from .cornet import cornet
    except Exception as e:
        raise e("Install cornet via pip install git+https://github.com/dicarlolab/CORnet")

    model = cornet.get_model(model_letter=model_letter, pretrained=pretrained, map_location=device)
    layers = cornet_pt_layers

    return model, layers


def get_vonenet(model_arch, pretrained=False, device='cuda', **kwargs):

    try:
        from .vonenet import vonenet
    except Exception as e:
        raise e("Install VOneNet via pip install git+https://github.com/dicarlolab/vonenet")

    url = f'https://vonenet-models.s3.us-east-2.amazonaws.com/{vonenet.FILE_WEIGHTS[model_arch.lower()]}'
    ckpt_data = torch.utils.model_zoo.load_url(url, map_location=device)

    stride = ckpt_data['flags']['stride']
    simple_channels = ckpt_data['flags']['simple_channels']
    complex_channels = ckpt_data['flags']['complex_channels']
    k_exc = ckpt_data['flags']['k_exc']

    noise_mode = ckpt_data['flags']['noise_mode']
    noise_scale = ckpt_data['flags']['noise_scale']
    noise_level = ckpt_data['flags']['noise_level']

    model_id = ckpt_data['flags']['arch'].replace('_', '').lower()

    model = vonenet.VOneNet(model_arch=model_id, stride=stride, k_exc=k_exc,
                            simple_channels=simple_channels, complex_channels=complex_channels,
                            noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)

    if pretrained:
        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = vonenet.Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

    vone_block_layers = [
        # 'vone_block.simple',
        # 'vone_block.complex',
        # 'vone_block.gabors',
        # 'vone_block.output',
        'bottleneck']

    alexnet_layers = ['features.1',
                      'features.4',
                      'features.6',
                      'features.8']

    vone_alexnet_pt_layers = vone_block_layers + [f"model.{layer}" for layer in alexnet_layers]
    vone_resnet50_pt_layers = vone_block_layers + [f"model.{layer}" for layer in resnet50_pt_layers]
    vone_cornet_pt_layers = vone_block_layers + ['model.V2.output', 'model.V4.output', 'model.IT.output']

    layers = None
    if 'resnet50' in model_arch:
        layers = vone_resnet50_pt_layers
    elif 'alexnet' in model_arch:
        layers = vone_alexnet_pt_layers
    elif 'cornets' in model_arch:
        layers = vone_cornet_pt_layers

    return model, layers


def get_efficientnet(name, pretrained=True):

    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', f'nvidia_{name}', pretrained=pretrained)

    layers = ['stem.activation', 'layers.0',
              'layers.1', 'layers.2', 'layers.3',
              'layers.4', 'layers.5', 'layers.6', 'features']

    return model, layers


def get_mobilenet(pretrained=True):

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)

    layers = [f'features.{i}' for i in range(8)]

    return model, layers


def get_densenet(name, pretrained=True):

    model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)

    layers = ['features.relu0',
              'features.denseblock1', 'features.transition1',
              'features.denseblock2', 'features.transition2',
              'features.denseblock3', 'features.transition3',
              'features.denseblock4']

    return model, layers


def get_wideresnet(name, pretrained=True):

    model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)

    if 'resnet50' in name:
        layers = resnet50_pt_layers
    else:
        layers = resnet101_pt_layers

    return model, layers


def get_barlowtwins(pretrained=True, device='cuda'):

    # https://github.com/facebookresearch/barlowtwins

    model = resnet50(weights=None)
    url = 'https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth'

    if pretrained:
        weights = torch.utils.model_zoo.load_url(url, file_name='barlowtwins_resnet50.pth', map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

        # # Pretrained barlowtwins readout weights for ImageNet
        # try:
        #     weights = torch.load(
        #         './simclr/checkpoint/barlowtwins_readout.pth')['model']
        #     weights = {key[len('module.'):]: val for key, val in weights.items()}
        #     missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        #     assert missing_keys == [] and unexpected_keys == []
        # except Exception as e:
        #     raise e

    layers = resnet50_pt_layers

    return model, layers


def get_SimCLR(model_name, pretrained=True, device='cuda'):

    import dimensionality.simclr as simclr

    # # Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
    # # All other options override the simclr_8node_resnet.yaml config.
    # https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md
    # https://github.com/facebookresearch/vissl/tree/main/configs/config/pretrain/simclr/models

    model_root = 'https://dl.fbaipublicfiles.com/vissl/model_zoo/'

    SimCLRresnet50w1_URL = 'simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch'
    SimCLRresnet50w4_URL = 'simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/model_final_checkpoint_phase999.torch'
    SimCLRresnet101_URL = 'simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch'

    simclr_dict = {'simclr_resnet50w1': dict(model=simclr.resnet.resnet50(width_multiplier=1),
                                             layers=resnet50_pt_layers,
                                             url=model_root+SimCLRresnet50w1_URL,
                                             filename='SimCLRresnet50w1_weights.torch'),
                   'simclr_resnet50w4': dict(model=simclr.resnet.resnet50(width_multiplier=4),
                                             layers=resnet50_pt_layers,
                                             url=model_root+SimCLRresnet50w4_URL,
                                             filename='SimCLRresnet50w4_weights.torch'),
                   'simclr_resnet101': dict(model=simclr.resnet.resnet101(width_multiplier=1),
                                            layers=resnet101_pt_layers,
                                            url=model_root+SimCLRresnet101_URL,
                                            filename='SimCLRresnet101_weights.torch'),
                   }

    model = simclr_dict[model_name]['model']

    weights = torch.utils.model_zoo.load_url(simclr_dict[model_name]['url'],
                                             file_name=simclr_dict[model_name]['filename'],
                                             map_location=device)

    prefix = "_feature_blocks."
    weights = {k[len(prefix):]: w for k, w in weights['classy_state_dict']['base_model']['model']['trunk'].items()}

    if pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

        # try:
        #     weights = torch.load(
        #         f'./simclr/checkpoint/{model_name}_readout.pth')['model']
        #     weights = {key[len('module.'):]: val for key, val in weights.items()}
        #     missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        #     assert missing_keys == [] and unexpected_keys == []
        # except Exception as e:
        #     raise e

    return model, simclr_dict[model_name]['layers']


def get_robustness(model_name, pretrained=True, device='cuda'):

    Resnet50_l2_3 = 'https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0'
    Resnet50_linf_4 = 'https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0'
    Resnet50_linf_8 = 'https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0'

    robustness_dict = {'robust_resnet50_l2_3': dict(model=resnet50(), layers=resnet50_pt_layers,
                                                    url=Resnet50_l2_3, filename='imagenet_l2_3_0.pt'),
                       'robust_resnet50_linf_4': dict(model=resnet50(), layers=resnet50_pt_layers,
                                                      url=Resnet50_linf_4, filename='imagenet_linf_4.pt'),
                       'robust_resnet50_linf_8': dict(model=resnet50(), layers=resnet50_pt_layers,
                                                      url=Resnet50_linf_8, filename='imagenet_linf_8.p')
                       }

    model = robustness_dict[model_name]['model']
    url = robustness_dict[model_name]['url']
    file_name = robustness_dict[model_name]['filename']
    chkp_path = os.path.join(torch.hub.get_dir(), f"checkpoints/{file_name}")

    if not os.path.exists(chkp_path):
        os.makedirs(os.path.dirname(chkp_path))
        os.system(f"wget -O {chkp_path} {url}")
    weights = torch.load(chkp_path, map_location=device)

    prefix = "module.model."
    model_weights = dict()
    for key, val in weights['model'].items():
        if prefix in key:
            key = key[len(prefix):]
            model_weights[key] = val
    weights = model_weights

    if pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(weights)
        assert missing_keys == [] and unexpected_keys == []

    return model, robustness_dict[model_name]['layers']


def get_moco_resnet50(pretrained=True):
    model = resnet50()
    layers = resnet50_pt_layers

    url = 'https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar'
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    head_dict = {}
    for key, val in state_dict['state_dict'].items():
        if 'module.' in key:
            key = key.split('module.')[-1]
            head_dict[key] = val

    if pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(head_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == []

    return model, layers
