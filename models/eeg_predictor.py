import torch
import torch.nn as nn
import numpy as np
from torchvision.models import alexnet
from src.torch_mpf import Torch_LayerwiseFWRF, Torch_FWRF
from src.torch_feature_space import fmapper, analyse_net, filter_dnn_feature_maps
from src.numpy_utility import iterate_range
from sklearn.decomposition import PCA

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

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

class pca_fmaps(nn.Module):
    def __init__(self, fmaps_fn, max_pc, stims, batchsize=100, device='cpu'):
        super(pca_fmaps, self).__init__()
        self.device = device
        self.fmaps_fn = fmaps_fn 
        ### produce fmaps data to calculate PC
        #print ('Calculating feature maps...')
        #_fmaps = fmaps_fn(torch.tensor(stims[:batchsize]).to(device))
        _fmaps = fmaps_fn(stims[:batchsize].clone().detach().to(device))
        all_fmaps = {k: [get_value(_fm),] for k,_fm in enumerate(_fmaps)}
        for rt,rl in iterate_range(batchsize, len(stims)-batchsize, batchsize):
            _fmaps = fmaps_fn(torch.tensor(stims[rt]).to(device))
            for k,_fm in enumerate(_fmaps):
                all_fmaps[k] += [get_value(_fm),]
        for k in all_fmaps.keys():
            all_fmaps[k] = np.concatenate(all_fmaps[k], axis=0)        
        
        ## calculate PCA object to apply
        self.fmaps_pca = {}
        self.m, self.s = {}, {}
        #print ('Calculating pca objects...')
        for k,fm in all_fmaps.items():
            self.fmaps_pca[k] = PCA(n_components=min(max_pc, np.prod(fm.shape[1:])))
            y = self.fmaps_pca[k].fit_transform((fm.reshape(len(fm), -1)))
            self.m[k] = torch.tensor(np.mean(y, axis=0, keepdims=True)).to(device)
            self.s[k] = torch.tensor(np.std(y, axis=0, keepdims=True)).to(device)
        
    def forward(self, _x): 
        _fmaps = self.fmaps_fn(_x)
        _pca_fmaps = []
        for k,pca in self.fmaps_pca.items():
            fm = get_value(_fmaps[k])
            _pca_fmaps += [torch.unsqueeze(torch.unsqueeze((torch.tensor(pca.transform((fm.reshape(len(fm), -1)))).to(self.device) - self.m[k])/self.s[k], 2), 3),]
        return _pca_fmaps  

class alexnet_layerwise_pcareg_eeg(nn.Module):
    def __init__(self, device, batch_size, data, nv=1, pre_nl=None, post_nl=None):
        super(alexnet_layerwise_pcareg_eeg, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        alexnet_model = alexnet(pretrained=True).to(device)
        # Freeze the parameters of the feature extractor
        for param in alexnet_model.parameters():
            param.requires_grad = False
            
        module_dict = analyse_net(alexnet_model, quiet=True)
        modules_list = [module for _, module in module_dict.items()]
        
        output_indices = [2, 4, 7, 9, 11, 17, 20, 21]
        self.feature_extractor = fmapper(modules_list, output_indices).to(device)
        
        self.feature_filter = None
        self.readout = None
        self.feature_filter, *_ = filter_dnn_feature_maps(data, self.feature_extractor, fmap_max=512, concatenate=False)
        
    def forward(self, x):
        _pca_fmaps_fn = pca_fmaps(self.feature_filter, 2, x, batchsize=self.batch_size, device=self.device)
        x=_pca_fmaps_fn(x)
        #x = self.feature_filter(x)
        self.readout = Torch_FWRF(x, rf_rez=1, nv=1, pre_nl=None, post_nl=None, dtype=np.float32).to(self.device)

        return self.readout(x)
        
