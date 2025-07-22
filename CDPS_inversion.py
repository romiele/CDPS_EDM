import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
from CDPS_utils import FaciesSet,compute_estimation_error

class CDPS_Inversion():
    def __init__(self, args):
        
        #load network
        with open(args.net_dir+args.net_snapfile, 'rb') as f:
            self.net = pickle.load(f)['ema'].to(args.device)
            
        self.work_dir = args.work_dir
        self.device = args.device
        self.sigma_max = args.sigma_max
        self.sigma_min = 0.002
        self.image_shp = args.image_shp
        self.rho = args.rho
        self.n_samples = args.n_samples
        self.num_steps = args.num_steps
        self.save_dir = args.save_dir
        self.hard_data = args.hard_data
        self.seismic = args.seismic
        self.data_obs_dir = args.data_obs_dir
        self.N_rmse = [0,0]
        
        #loading target data (comment this for different applications when you want to load dobs rather than calculate synthetic dobs from target)---------------------------------------------------------------
        self.target = torch.zeros(self.image_shp).to(self.device).double()
        
        for i in range(2): self.target[i] = torch.load(args.data_obs_dir + args.data_obs[i], weights_only=True, map_location=self.device).double()
            
        self.ip= [self.target[1].min(),self.target[1].max(), self.target[1].max() - self.target[1].min()]
        
        #padding is required because 100 is irregular for diffusion model
        if self.image_shp[-1]==100: self.target= self.pad(self.target)
        
        
        #conditioning on seismic data----------------------------------------------------
        if self.seismic: 
            self.load_wavelet_define_conv(args)
            
            #comment below if dobs file is available
            self.ys_obs = self.physics_forward(self.target[1][None,None,:])
            
            #decomment below if dobs file is available
            #self.ys_obs = torch.load(args.data_obs_dir + args.data_obs[1], weights_only=True, map_location=self.device).double()
            
            #define jacobian
            self.conv_jacobian = self.conv_matrix_J().to(self.device)
            
            #Noise sigma
            self.sigma_ys = args.seismic_data_error[1]*torch.abs(self.ys_obs.detach().clone())+args.seismic_data_error[0]
            self.var_ys = self.sigma_ys**2
        
            #contaminate with noise (comment below if dobs file is available) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.noise_ys = torch.randn_like(self.ys_obs)*self.sigma_ys
            self.ys_obs = (self.ys_obs + self.noise_ys)
            torch.save(self.noise_ys, self.save_dir+'/noise_ys.pt')
            if self.image_shp[-1]==100: #ignore everything in padded area
                self.ys_obs[...,100:] = 0; self.sigma_ys[...,100:] = 0; self.var_ys[...,100:] = 0
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                
            self.yslims= [self.ys_obs.min(), self.ys_obs.max()]
            self.N_rmse[1] = int(np.prod(self.image_shp[1:]))
            
        #conditioning on hard data-------------------------------------------------------
        if self.hard_data:
            
            #decomment below if dobs file is available
            #self.yh_obs = torch.load(args.data_obs_dir + args.data_obs[0], weights_only=True, map_location=self.device).double()

            #comment below if dobs file is available # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.mask = torch.zeros(self.image_shp).to(self.device)
            if self.image_shp[-1]==100: self.mask = self.pad(self.mask)
            for i in args.hd_cond_where:
                if i==30: self.mask[...,:40,:100] += torch.flip(torch.diag(torch.ones(100),15)[...,:40,:100], [1]).to(self.device)
                else: self.mask[...,i] += 1
            self.mask[self.mask>1]=1
            
            #decomment below if dobs file is available (hard data should be sparse and = 0 where not available (Ip))
            #self.mask = torch.zeros(self.image_shp).to(self.device) 
            #self.mask[:,self.yh_obs[1]!=0] = 1
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            self.yh_obs = self.mask*self.target[0].squeeze()
            self.yh_obs[1] = self.mask[1]*((self.target[1]-self.ip[0])/self.ip[2]).squeeze()
            self.sigma_yh = torch.zeros_like(self.yh_obs).to(self.device)
            self.sigma_yh[0] = args.hard_data_error[0]; self.sigma_yh[1] = args.hard_data_error[1]
            torch.save(self.yh_obs, self.save_dir+'/yh_obs.pt')

            #contaminate with noise (comment below if dobs file is available) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.noise_yh = torch.randn_like(self.yh_obs)*self.sigma_yh*self.mask ; self.yh_obs+=self.noise_yh
            torch.save(self.yh_obs, self.save_dir+'/noise_yh.pt')
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            self.N_rmse[0] = torch.sum(self.mask[0]).item()

        return None
        
    
    def pad (self,array):  
        return F.pad(array, (0, 4, 0, 0))
    
    
    def load_wavelet_define_conv(self, args):
        # wavelet = np.genfromtxt(args.data_obs_dir + args.wavelet_file) #Decomment here if you want to use the wavelet file
        
        # Comment below if you want to use wavelet file
        frequency = 1/(torch.pi*5)
        t= torch.arange(-20,20,1)
        omega = torch.pi * frequency
        wavelet = ((1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2))*100
        
        wavelet = np.expand_dims(wavelet, 0) # add bacth [B x H x W x C]
        if wavelet.ndim==2: 
            wavelet= np.expand_dims(wavelet, 0)
            wavelet= np.expand_dims(wavelet, -1)
        # # # # # #
        
        self.wavelet = torch.from_numpy(wavelet).double().to(args.device)
        k = self.wavelet.shape[-2]
        self.padding = (k//2,0)
        self.seismic_conv = torch.nn.Conv2d(1,1, kernel_size=1, padding= self.padding, bias=False)
        self.seismic_conv.weight = torch.nn.Parameter(self.wavelet).requires_grad_(False)
        self.len_wav = len(self.wavelet.squeeze())
        return None


    def physics_forward(self, realization):
        ip = torch.cat((realization, realization[:,:,[-1],:]), dim=2) # repeats last element
        ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
        ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
        rc = ip_d / ip_a
        return self.seismic_conv(rc)[:,:,:self.image_shp[1],:]


    def diag_els (self, xi, xii):
        #Jacobian of reflectivity coefficients  models
        den = (xi+xii)**2
        num1 = -2*xii
        num2 = 2*xi
        return torch.tensor([num1/den, num2/den])


    def conv_matrix_J(self):
        #creates the matrix for 1D convolution, input data is padded
        #same as torch, but torch is more efficient at forward, this is needed for Jacobian 
        wavelet_lenght = len(self.wavelet.squeeze())
        trace_lenght = self.image_shp[1]+wavelet_lenght
        toeplitz_w = torch.zeros(trace_lenght-wavelet_lenght+1-wavelet_lenght%2, 
                                 trace_lenght) 
        
        for i in range(toeplitz_w.shape[0]):
            toeplitz_w[i,i:i+wavelet_lenght]=self.wavelet.squeeze()
            
        return toeplitz_w


    def cond_grad_HD(self, x_0hat, x_t, i = 0):
        
        grads = torch.zeros_like(x_0hat.detach())
        wrmse = torch.zeros(self.minibatch)
        covar = torch.zeros_like(self.sigma_yh).to(self.device)
        
        x_0hat_r = (x_0hat.squeeze()+1)/2
        d_hat = x_0hat_r*self.mask
        difference = ((self.yh_obs - d_hat)**2)

        if args.method == 'cdps':
            #covariance is the noise in the observed data + the noise in the image itself (both Gaussian uncorrelated so just add diagoanl matrices)
            for p in range(self.image_shp[0]):
                covar[p] = self.sigma_yh[p]**2 + self.sigma_xhat0[i][p]**2

        elif args.method == 'dps':
            #covariance is the noise in the observed data 
            covar = self.sigma_yh**2

        if args.method != 'actual_dps':
            #Maximum likelihood for Gaussian Noise (not implemented for other types)
            err = (difference/covar)
            norm = - .5 * torch.sum(err.flatten(1), dim=1)
        
        else:
            #this is the way they actually implement it in Chung et al. 2022 for DPS inversion (not correct)
            norm = - torch.linalg.norm(difference)[None,]
            
        for i in range(self.minibatch): 
            #loop is required for gradient caluclation independent for each sample
            grads[i] = torch.autograd.grad(outputs=norm[i], inputs=x_t, retain_graph=True)[0][i].detach() #gradients        
            wrmse[i] = ((difference/(self.sigma_yh**2)).sum()/self.N_rmse[0]).sqrt().detach().cpu()
            
            torch.cuda.empty_cache()
            
        return grads, wrmse


    def cond_grad_nonlinear(self, x_0hat,x_t,i=0):
        
        #compute d_hat and error from observed
        grads = torch.zeros_like(x_0hat.detach())
        wrmse = torch.zeros(self.minibatch)
        norm = torch.zeros(self.minibatch)
        
        x_0hat_r = (x_0hat+1)/2
        x_0hat_r_ip = ((x_0hat_r[:,1,None])*self.ip[2]+self.ip[0])
        d_hat = self.physics_forward(x_0hat_r_ip)
        difference = (d_hat - self.ys_obs)
        
        if args.method == 'cdps':
            assert(self.minibatch==1), 'Not implemented for > 1 realization at a time'
            
            #take the sigma value for xhat0 at time t_hat - rescaled to Ip support
            sigma_x0 = self.sigma_xhat0[i][1]*self.ip[2]
        
            #get the corresponding covariance matrix for Gaussian uncorrelated noise (d x d, d=len(seismic trace))
            temp_eye = torch.eye(self.image_shp[-2]).double().to(self.device)*(sigma_x0**2)
            
            #Jacobian structure for RC (changes value at each trace)
            JRC_struct = torch.zeros(self.conv_jacobian.shape[1],self.conv_jacobian.shape[1],).to(self.device)
            
            #inverse of the full covariances are saved to maintain autograd stable
            inv_cov = torch.zeros(self.image_shp[-1], self.image_shp[-2], self.image_shp[-2]).to(self.device)
            norm_t = torch.zeros(self.image_shp[-1])

            for t in range(self.image_shp[-1]):
                trace = x_0hat_r_ip.squeeze()[:,t]
                
                # if 2nd is trace[i], assume last element is repeated
                # idx = self.len_wav//2-1
                # JRC_struct[idx,idx:idx+2]= self.diag_els(trace[0], trace[0]) 
                for i in range(self.image_shp[-2]-1):
                    idx = i+self.len_wav//2
                    JRC_struct[idx,idx:idx+2] = self.diag_els(trace[i], trace[i+1])
                idx = i+1+self.len_wav//2
                JRC_struct[idx,idx]= self.diag_els(trace[i], 0)[0] # if 2nd is trace[i], assume last element is repeated

                J_trace = (self.conv_jacobian @ JRC_struct)[:-1, self.len_wav//2:-self.len_wav//2].double()
            
                #Full covariance matrix = Error propagation + observed data error
                cov = torch.matmul(J_trace.T, torch.matmul(temp_eye, J_trace))
                cov += torch.diag(self.var_ys[0,0,:,t]).to(self.device)
                
                inv_cov[t]  = torch.inverse(cov) 
                norm_t[t] = - .5*torch.matmul(difference[0,0,:,t], torch.matmul(inv_cov[t].double(),  difference[0,0,:,t]))

            norm = torch.sum(norm_t)
            
        else:
            err = ((difference)[...,:100]**2/(self.var_ys)[...,:100])
            norm = - .5* torch.sum(err.flatten(1), dim=1)

        grads = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0].detach()
        
        difference.detach_()
        wrmse = ((((difference/self.sigma_ys)**2)[...,:100]).sum()/self.N_rmse[1]).sqrt().detach().cpu()

        torch.cuda.empty_cache()
        
        return grads, wrmse
    
    
    def condition(self, denoised, x_hat, i):
        cond_grads = torch.zeros_like(denoised)
        rmset= torch.zeros(self.minibatch, 2)
        if self.hard_data: 
            gradients, rmse= self.cond_grad_HD(denoised, x_hat, i)      #calculate the gradients and loss
            cond_grads+= gradients
            rmset[:,0]= rmse
            
        if self.seismic: 
            gradients, rmse= self.cond_grad_nonlinear(denoised, x_hat, i)          #calculate the gradients and loss
            cond_grads+= gradients
            rmset[:,1]=rmse
        return cond_grads, rmset
    
        
    def edm_inverse_sampler(self, 
                    n_samples, class_labels=None, randn_like=torch.randn_like,
                     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):  
                     # S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):  
        """
            This is a modified edm sampler to include conditioning to hard data and seismic
        """
        
        self.minibatch = n_samples
        if self.seismic: wrmse_s_t = torch.zeros(self.minibatch, self.num_steps)
        if self.hard_data: wrmse_h_t = torch.zeros(self.minibatch, self.num_steps)
        
        temp_shp = self.image_shp.copy(); temp_shp.insert(0,self.minibatch)
        latents = torch.randn(temp_shp).to(self.device)
        
        if self.image_shp[-1]==100: latents = self.pad(latents)
        
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        # Main sampling loop.
        text = f"{'Seismic' if self.seismic else ''} {'Hard data' if self.hard_data else ''}"
        desc = f'Condintioning on {text} - {self.minibatch} samples, {self.num_steps} steps' if (self.seismic or self.hard_data) else None
        x_next = latents.to(torch.float64) * t_steps[0]
        
        pbar = tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), desc= desc)
        for i, (t_cur, t_next) in pbar: # 0, ..., N-1
            
            x_cur = x_next
            
            # Increase noise temporarily.
            gamma = min(S_churn / self.num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            
            #evaluate
            x_hat.requires_grad_(True)
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64) #get xhat_0|x_t
            cond_grads, wrmse = self.condition(denoised, x_hat,i) #get physics gradients dL/dx_t
            
            #preserve memory
            denoised.detach_(); x_hat.detach_(); x_next.detach_(); torch.cuda.empty_cache()
            
            #compute next step (Euler)
            score_func = (denoised-x_hat)/(t_hat**2)
            if args.method == 'actual_dps': #their way, adapted to EDM
                d_cur = - t_hat * (score_func)
                x_next = x_hat + (t_next - t_hat) * d_cur - cond_grads
                
            else:
                d_cur = - t_hat * (score_func + cond_grads)
                x_next = x_hat + (t_next - t_hat) * d_cur
            
            if args.method != 'actual_dps': #let's just skip this is not necessary
                # Apply 2nd order correction.
                if i < self.num_steps - 1:
                    
                    #evaluate
                    x_next.requires_grad_(True)
                    denoised = self.net(x_next, t_next.unsqueeze(0), class_labels).to(torch.float64) #get xhat_0|x_next
                    cond_grads, wrmse = self.condition(denoised, x_next,i) #get physics gradients dL/dx_next
                    
                    #preserve memory
                    denoised.detach_(); x_hat.detach_(); x_next.detach_(); torch.cuda.empty_cache()
                    
                    #compute next step (2nd order)
                    score_func = (denoised-x_hat)/(t_hat**2)
                    d_prime = - t_hat * (score_func + cond_grads)
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
            pbar.set_postfix({'WRMSE': wrmse}, refresh=False)
        
            #store rmse for logging progress
            if self.hard_data: wrmse_h_t[:,i] = wrmse[:,0]
            if self.seismic: wrmse_s_t[:,i] = wrmse[:,1]

        #storing stuff for safety
        try:
            temp_real = torch.load(self.save_dir+'/realizations_temp.pt', map_location='cpu')
            if self.hard_data: temp_wrmse_h = torch.load(self.save_dir+'/wrmse_h_temp.pt', weihgts_only = True, map_location='cpu')
            if self.seismic: temp_wrmse_s = torch.load(self.save_dir+'/wrmse_s_temp.pt', weihgts_only = True, map_location='cpu')
            
            torch.save(torch.cat((temp_real, x_next.detach().cpu()), dim=0), self.save_dir+'/realizations_temp.pt')
            if self.hard_data: torch.save(torch.cat((temp_wrmse_h, wrmse_h_t.detach().cpu()), dim=0), self.save_dir+'/wrmse_h_temp.pt')
            if self.seismic: torch.save(torch.cat((temp_wrmse_s, wrmse_s_t.detach().cpu()), dim=0), self.save_dir+'/wrmse_s_temp.pt')
        
        except Exception as error:
            print(error)
            torch.save(x_next.detach().cpu(), self.save_dir+'/realizations_temp.pt')
            if self.hard_data: torch.save(wrmse_h_t.detach().cpu(), self.save_dir+'/wrmse_h_temp.pt')
            if self.seismic: torch.save( wrmse_s_t.detach().cpu(), self.save_dir+'/wrmse_s_temp.pt')
            
        torch.cuda.empty_cache()
        time.sleep(1)
        del score_func, denoised,x_hat,latents
        if (self.seismic and not self.hard_data) : 
            return x_next, (torch.nan, wrmse_s_t)
        elif (not self.seismic and self.hard_data) : 
            return x_next, (wrmse_h_t, torch.nan)
        elif (self.seismic and self.hard_data): 
            return x_next, (wrmse_h_t, wrmse_s_t)
    
    
    def invert(self):
        temp = self.image_shp.copy()
        temp.insert(0,self.n_samples)
        temp[-1] = temp[-1] if temp[-1]!=100 else temp[-1]+4 
        realizations=torch.zeros(temp)
        wrmse_s = torch.zeros(self.n_samples,self.num_steps)
        wrmse_h = torch.zeros(self.n_samples,self.num_steps)
        
        for i in range(self.n_samples):
            realizations_t, wrmses = self.edm_inverse_sampler(1)
            
            if self.seismic : wrmse_s[i,:] = wrmses[0]
            if self.hard_data : wrmse_h[i,:] = wrmses[1]

            realizations[i]=realizations_t.detach().cpu()
            
        realizations = (realizations+1)/2
        realizations[:,1] = (realizations[:,1])*(self.ip[1].item()-self.ip[0].item())+self.ip[0].item()
        realizations = realizations.detach().cpu().squeeze()

        torch.save(realizations, self.save_dir+'/reals.pt')
        torch.save(wrmse_s, self.save_dir+'/wrmse_s.pt')
        torch.save(wrmse_h, self.save_dir+'/wrmse_h.pt')
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Project directories
    parser.add_argument("--net_dir",             default= '/',                                 help = '(Valid for this test code) Directory where diffusion model is stored') 
    parser.add_argument("--net_snapfile",        default= '/network-snapshot-001800.pkl',      help = '(Valid for this test code) Snapshot/Checkpoint of the trained network') 
    
    parser.add_argument("--work_dir",            default= '/',                                 help = 'Working directory')
    parser.add_argument("--save_dir",            default= '/',                                 help = 'Parent directory where to save results')
    parser.add_argument("--train_data_dir",      default= '/',                                 help = 'If evaluating the denoising error, training data folder is required')
    parser.add_argument("--data_obs_dir",        default= '.../test_models/',                  help = '(For this specific application) Where test models (true Facies and Ip) are stored\
                                                                                                             (Otherwise) Where conditioning data is stored')
    # Conditioning data files
    parser.add_argument("--data_obs",            default= ['mtest_facies1.pt', 'mtest_ip1.pt'], help = '(For this specific application) Target distributions [true Facies and Ip] \
                                                                                                        (Otherwise) Conditioning data in data_obs_dir [HD, Seismic]')
    #Hard data conditioning
    parser.add_argument("--hard_data",           default = True,                               help = 'True if conditioning on hard data / well logs')
    parser.add_argument("--hd_cond_where",       default = [50,30],                            help = '(For this specific application) x locations of the conditioning well logs\
                                                                                                      (otherwise) Remove: observed data should be stored in a file and loaded as masked data')
    parser.add_argument("--hard_data_error",     default = [0.1,0.05],                         help = 'Standard deviation values for [Facies, Ip], considering the normalized properties in range [0,1[')
                                                                                                      
    parser.add_argument("--seismic",             default = True,                               help = 'True if conditioning on seismic data')             
    parser.add_argument("--wavelet_file",        default = 'wavelet.asc',                      help = 'If seismic data conditioning, wavelet file') 
    parser.add_argument("--seismic_data_error",  default = [1,.05],                            help = '[Absolute, Relative] components of Gaussian error in seismic data')
    
    #Modeling parameters
    parser.add_argument("--n_samples",           default = 1,                                  help = 'Number of samples to generate')
    parser.add_argument("--num_steps",           default = 32,                                 help = 'Number of denoising steps to generate a realization (minimum required suggested: linear=32; nonlinear=250)')
    
    parser.add_argument("--device",              default = "cuda:0",                           help = 'CPU/GPU Device')
    parser.add_argument("--image_shp",           default = [2,80,100],                         help = '[n. of properties, y, x] of generated realizations')
    parser.add_argument("--sigma_max",           default = 80,                                 help = 'Max noise in noisy vector')
    parser.add_argument("--rho",                 default = 7,                                  help = 'Determines the noise schedule, 1 is linear. 7 is the optimal exponential trend from Karras EDM paper')  
         
    parser.add_argument("--method",              default = 'cdps',                             help = 'cdps / dps / actual_dps; leave it as "cdps" for proposed algorithm\
                                                                                                       (for additional tests: \
                                                                                                          "dps" is the conventional way it is theoretically proposed \
                                                                                                          "actual_dps" original implementation using RMSE for likelihood function (weight = 1)')
    args = parser.parse_args()
    
##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% ##### %%%%% #####
    
    sys.path.append(args.net_dir) #For EDM, it is necessary to use dnnlib folder from the original code
    
    CDPS = CDPS_Inversion(args)
    # using training data to compute the error
    if args.method=='cdps':
        try: 
            print('Attempting to load diffusion model error')
            CDPS.sigma_xhat0 = torch.load(args.work_dir+f'/xhat0_sigma_{args.sigma_max}_{args.rho}_{args.num_steps}.pt', 
                                     weights_only=True, map_location=args.device)
        except: 
            print('Diffusion model error not found, estimating model\'s error')
            dataset = FaciesSet(args.train_data_dir, [args.image_shp[1],args.image_shp[2]], args.image_shp[0])
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
            for i, test_data in enumerate(loader):
                test_data = test_data[0]
                break
            CDPS = compute_estimation_error(CDPS,test_data.to(args.device))
            
    text = 'Both' if (args.seismic and args.hard_data) else ('Seismic' if args.seismic else 'HD')
    args.save_dir = args.save_dir+f'/Inversion_{args.method}_On{text}'

    os.makedirs(args.save_dir, exist_ok=True)
    
    CDPS.invert()
            
