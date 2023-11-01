import torch
import torch.nn as nn
import numpy as np
from torchvision.models import alexnet
from src.torch_mpf import Torch_LayerwiseFWRF, Torch_FWRF
from src.torch_feature_space import fmapper, analyse_net, filter_dnn_feature_maps

class alexnet_layerwiseFWRF_eeg(nn.Module):
    def __init__(self, device, nv=1, pre_nl=None, post_nl=None):
        super(alexnet_layerwiseFWRF_eeg, self).__init__()
        
        alexnet_model = alexnet(pretrained=True).to(device)
        # Freeze the parameters of the feature extractor
        for param in alexnet_model.parameters():
            param.requires_grad = False
            
        module_dict = analyse_net(alexnet_model, quiet=True)
        modules_list = [module for _, module in module_dict.items()]
        
        output_indices = [2, 4, 7, 9, 11, 17, 20, 21]
        self.feature_extractor = fmapper(modules_list, output_indices)

        
        # The readout layer will be initialized after we get the feature map shapes
        self.readout = None
        self.nv = nv
        self.pre_nl = pre_nl
        self.post_nl = post_nl

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.readout is None:  # Initialize the readout layer once we have the feature map shapes
            self.readout = Torch_LayerwiseFWRF(features, nv=self.nv, pre_nl=self.pre_nl, post_nl=self.post_nl).to(x.device)
        return self.readout(features)


class alexnet_layerwise_pcareg_eeg(nn.Module):
    def __init__(self, device, nv=1, pre_nl=None, post_nl=None):
        super(alexnet_layerwise_pcareg_eeg, self).__init__()
        
        alexnet_model = alexnet(pretrained=True).to(device)
        # Freeze the parameters of the feature extractor
        for param in alexnet_model.parameters():
            param.requires_grad = False
            
        module_dict = analyse_net(alexnet_model, quiet=True)
        modules_list = [module for _, module in module_dict.items()]
        
        output_indices = [2, 4, 7, 9, 11, 17, 20, 21]
        self.feature_extractor = fmapper(modules_list, output_indices)
        #self.feature_filter, lmask, fmask = filter_dnn_feature_maps(data, _fmaps_fn, batch_size, fmap_max=1024, concatenate=True, trn_size=None):

        
        # The readout layer will be initialized after we get the feature map shapes
        self.readout = None
        self.nv = nv
        self.pre_nl = pre_nl
        self.post_nl = post_nl

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.readout is None:  # Initialize the readout layer once we have the feature map shapes
            self.readout = Torch_LayerwiseFWRF(features, nv=self.nv, pre_nl=self.pre_nl, post_nl=self.post_nl).to(x.device)
        return self.readout(features)
