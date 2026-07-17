!! This code has been substatially updated: a more generalized and potentially efficient version is currently being depoloyed in https://github.com/romiele/CDPS. The CDPS repository will be updated as my research on the topic will continue, eventually integrating different interpolators (DDIM/DDPM, EDM, flow map matching etc) and geophysical conditioning methods (now including tomograms downscaling and probabilistic inversion of Electrical apparent resistivity profiles).

This repository contains the CDPS for score-based diffusion models from "Diffusion models for multivariate subsurface generation and efficient probabilistic inversion", in https://doi.org/10.1016/j.cageo.2025.106076 (with a small corrigendum here: https://doi.org/10.1016/j.cageo.2026.106138) or https://arxiv.org/abs/2507.15809 (integrating correction). 
The code is ready to work with the data in "Test models", for conditioning using Well logs and Seismic data. 
The main file, CDPS_inversion.py is to be used, fill in the parameters within this file and run for inversion. 

Wether you need any support or question, contact me at roberto.miele@unil.ch , I'll be happy to help.

Examples of inversion with CDPS
![Description of gif](/inversion_animation.gif)



Currently, the CDPS for DDIM and DDPM, and the training dataset for the Diffusion model are available under request (Again, contact me at roberto.miele@unil.ch)


