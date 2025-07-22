# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from tqdm import tqdm

# try:
#     import pyspng
# except:
#     pyspng = None
pyspng = None

from torchvision import transforms
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        **ignore                # I am ignoring "resolution" key in training_loop, seems like a bug 
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        try:
            assert self.image_shape[1] == self.image_shape[2]
        except: 
            print(f'Irregular image of shape {self.image_shape}, {self.image_shape[1]} defines resolution')
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


#Datasetobj that loads facies and Ip distributions
class FaciesSet(Dataset):

    def __init__(self,
                 path,
                 image_size,
                 image_depth,
                 **super_kwargs 
                 ):
        
        t = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomInvert(1),
                transforms.Normalize(0.5,0.5),
                transforms.RandomVerticalFlip(1),
                ])
    
        self.path = path
        self.crop = False if image_size[0] in [80,100,128] else True
        
        #self.resolution = image_size[0] #for irregularly shaped images, use smaller resolution
        self.size = image_size
        self.len_data = len(os.listdir(self.path+'/Facies/'))
        self.transform = t
        self.multi= True if image_depth>1 else False
        if self.multi:
            self.max_ip_reals=0 ; i=0; t = True
            while t==True:
                t = os.path.isfile(self.path+f'/Ip/0_{i}.pt')
                if t==True: self.max_ip_reals +=1
                i+=1
            ip1 = torch.load(self.path+f'/Ip/{np.random.randint(0,self.len_data)}_0.pt', weights_only=True)
            ip2 = torch.load(self.path+f'/Ip/{np.random.randint(0,self.len_data)}_0.pt', weights_only=True)

            self.ipmin = min(ip1.min(),ip2.min())
            self.ipmax = max(ip1.max(),ip2.max())
            
            if self.crop: ip1= ip1[:,:self.size[0],:self.size[1]]
            
        name = os.path.splitext(os.path.basename(self.path))[0]
        raw_shape = [self.len_data] + [1 if not self.multi else 2] + list(ip1.squeeze().shape)
        
        #if image_size is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #    raise IOError('Image files do not match the specified resolution')
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        facies_name = self.path+f'/Facies/{idx}.png'
            
        facies = self.transform(PIL.Image.open(facies_name))
        
        #random sample a reandom 64x64 block in the 80x100 images
        if self.crop==True:
            z= np.random.randint(0,80-self.size[0]) 
            x= np.random.randint(0,100-self.size[1])
            facies = facies[0,None,z:z+self.size[0],x:x+self.size[1]]
        
        else: 
            facies = facies[0,None,:]
        
        # facies= facies*2-1
        out_dict = torch.zeros(0) #this is just to integrate possible conditions in a second time
        
        if self.multi:
            Ip = self.path+f'/Ip/{idx}_{np.random.randint(0,self.max_ip_reals)}.pt'
            
            Ip = torch.load(Ip, weights_only=True)
            
            if self.crop==True: Ip = Ip[0,None,z:z+self.size[0],x:x+self.size[1]]
            
            Ip = 2*((Ip - self.ipmin) / (self.ipmax - self.ipmin))-1
            
            # Padding is for images having size x = 100 
            if 100 in self.image_shape:
                out = F.pad(torch.cat((facies,Ip), dim=0), (0, 4, 0, 0))
            
            else:
                out = torch.cat((facies,Ip), dim=0)

            return out, out_dict
            
        else:
            if 100 in self.image_shape:
                out = F.pad(torch.cat((facies,Ip), dim=0), (0, 4, 0, 0))
            else: out = facies.detach()
            
            return out, out_dict
#----------------------------------------------------------------------------

def compute_estimation_error(CDPS, test_data):
    # define the estimation errors at each timestep, individually from a set of test data
    try: 
        CDPS.sigma_xhat0 = torch.load(CDPS.work_dir+f'/xhat0_sigma_{CDPS.sigma_max}_{CDPS.rho}_{CDPS.num_steps}.pt', weights_only=True, map_location=CDPS.device)
    except: 
        N_ex = test_data.shape[0]
        div = 4 if N_ex > 4 else 1
        bs = int(N_ex/div) #batch size
        CDPS.sigma_xhat0=torch.zeros(CDPS.num_steps+1, test_data.shape[1]).to(CDPS.device)
        
        # define the sigma sampling schedule 
        # Adjust noise levels based on what's supported by the CDPS.
        sigma_min = max(CDPS.sigma_min, CDPS.net.sigma_min)
        sigma_max = min(CDPS.sigma_max, CDPS.net.sigma_max)
        
        # Time step discretization.
        step_indices = torch.arange(CDPS.num_steps, dtype=torch.float64, device=CDPS.device)
        t_steps = (sigma_max ** (1 / CDPS.rho) + step_indices / (CDPS.num_steps - 1) * (sigma_min ** (1 / CDPS.rho) - sigma_max ** (1 / CDPS.rho))) ** CDPS.rho
        
        for i in tqdm(range(len(t_steps)), desc= 'Error at time-step'):
            #error vectors
            err_y = torch.randn_like(test_data).to(CDPS.device)*t_steps[i]
            residuals = torch.zeros_like(test_data)
            #estimate x_0 at specific error
            D_yn = torch.zeros_like(test_data)
            for j in range(div):
                D_yn[j*bs:j*bs+bs] = CDPS.net(test_data[j*bs:j*bs+bs] + err_y[j*bs:j*bs+bs], t_steps[i], None)
                residuals[j*bs:j*bs+bs] = (test_data[j*bs:j*bs+bs]-D_yn[j*bs:j*bs+bs])
            if j*bs+bs != N_ex:
                D_yn[j*bs+bs:] = CDPS.net(test_data[j*bs+bs:] + err_y[j*bs+bs:], t_steps[i], None)
                residuals[j*bs+bs:] = (test_data[j*bs+bs:]-D_yn[j*bs+bs:])

            #compute the residual for different images at noise level t
            mean_err_pixel = torch.sqrt(torch.mean((residuals.flatten(2)/2)**2, dim=0))
            CDPS.sigma_xhat0[i,:] = torch.quantile(mean_err_pixel.flatten(1), q=0.5, dim=1)
        torch.save(CDPS.sigma_xhat0, CDPS.work_dir+f'/xhat0_sigma_{CDPS.sigma_max}_{CDPS.rho}_{CDPS.num_steps}.pt')

    return CDPS