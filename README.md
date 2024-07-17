# Project Title and Description

Title : Diffusion Models for Gravitational Lensing Simulation

Description : Strong gravitational lensing is a promising probe of the substructure of dark matter to better understand its underlying nature. Deep learning methods have the potential to accurately identify images containing substructure and differentiate WIMP particle dark matter from other well-motivated models, including axions and axion-like particles, warm dark matter, etc.

Traditional simulations of gravitational lensing are time-consuming and require extensive computational resources. This project proposes the use of diffusion models, a class of generative models known for their ability to produce high-quality, detailed images from a distribution of noise, to simulate strong gravitational lensing images. We aim to generate realistic simulations of gravitational lensing events that can be used to augment datasets for machine learning models and facilitate the development of better domain adaptation and self-supervised models aimed at bridging the gap between simulated and real images of gravitational lensing. Furthermore, we will also investigate leveraging conditional diffusion models to generate gravitational lensing simulations by conditioning the model on specific parameters related to the lensing events, such as the mass distribution of the lensing galaxy, orientation, and the redshift of both the source and the lens.

# Project Structure

-- cfg  
    |--- ddpm_config.yaml <- configuration file for DDPM   
    |--- ae_md_cond_ddpm_config.yaml <- configuration file for using Autoencoder and DDPM   
    |--- ae_md_config.yaml <- configuration file for using Autoencoder and DDPM   
    |--- conditional_ddpm_config <- configuration file for using all variables       
-- dataset     
    |--- preprocessing_model_2.py <- data preprocessing for model 2   .   
    |--- preprocessing_md_model2.py <- data preprocessing for mass distribution  
    |--- preprocessing_all_model2.py <- data preprocessing for all variables    
-- model    
    |--- ddpm.py <- diffusion model parameters    
    |--- unet_sa.py <- unet architecture with self attention     
    |--- autoencoder.py <- autoencoder architecture  
    |--- clip.py <- clip mechanicm implementation  
    |--- ddpm_all.py <- diffusion model for all variables  
    |--- resnet.py <- resnet18 architecture  
    |--- unet_all <- unet for all variables  
    |--- unnet_sa <- unet for all variables adding variable embedding separetely  
-- plots     
    |--- saving images     
-- saved_models    
    |--- save trained models     
-- scripts   
    |--- run_ae_ddpm.py <- script to run autoencoder and ddpm  
    |--- run_ae.py <- script to run autoencoder  
    |--- run_all.py <- script to run ddpm for all variables   
    |--- run_conditional_ddpm.py <- script to run conditional ddpm   
    |--- run_ddpm.py <- script to run ddpm   
    |--- run_fid.py <- script to run fid score  
    |--- run_reg_resnet_all.py <- script to run resnet18 for all variables  
    |--- run_resnet_regression.py <- script to run resnet18 for regression  
    |--- run_resnet.py <- script to run resnet18  
    |--- run_roc.py <- script to run roc plot       
-- train      
    |--- train_ae_ddpm.py <- training loop for autoencoder and ddpm  
    |--- train_ae.py <- training loop for autoencoder  
    |--- train_all.py <- training loop for all variables  
    |--- train_conditional_ddpm.py <- training loop for conditional ddpm  
    |--- train_ddpm.py <- training loop for ddpm  
    |--- train_resnet.py <- training loop for resnet     
-- utils      
    |--- test.ipynb <- testing commands    
    |--- test.py <-   
    |--- plot_all <- plotting script  
    |--- plot_ae_ddpm <- plotting script when using autoencoder and ddpm  


