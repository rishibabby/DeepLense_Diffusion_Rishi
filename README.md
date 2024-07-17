# Project Title and Description

Title : Diffusion Models for Gravitational Lensing Simulation

Description : Strong gravitational lensing is a promising probe of the substructure of dark matter to better understand its underlying nature. Deep learning methods have the potential to accurately identify images containing substructure and differentiate WIMP particle dark matter from other well-motivated models, including axions and axion-like particles, warm dark matter, etc.

Traditional simulations of gravitational lensing are time-consuming and require extensive computational resources. This project proposes the use of diffusion models, a class of generative models known for their ability to produce high-quality, detailed images from a distribution of noise, to simulate strong gravitational lensing images. We aim to generate realistic simulations of gravitational lensing events that can be used to augment datasets for machine learning models and facilitate the development of better domain adaptation and self-supervised models aimed at bridging the gap between simulated and real images of gravitational lensing. Furthermore, we will also investigate leveraging conditional diffusion models to generate gravitational lensing simulations by conditioning the model on specific parameters related to the lensing events, such as the mass distribution of the lensing galaxy, orientation, and the redshift of both the source and the lens.

# Project Structure
<pre>

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

</pre>


# Class Conditional Diffusion model

I have trained a diffusion model using labels as a condition, with the specific labels being "axion," "cdm," and "no_sub." In addition, I have trained a ResNet model on these same classes. To evaluate the performance, I generated images using the trained diffusion model and then used these images to plot a Receiver Operating Characteristic (ROC) curve. This allowed me to assess the effectiveness of the ResNet model in distinguishing between the different classes based on the generated images.

![alt text](image.png)

# Variable Conditional Diffusion model

I have trained a diffusion model conditioned on all astrophysical variables, alongside training a ResNet-18 model using these variables for regression. Following the training, I generated images using the diffusion model and created scatter plots to visualize the relationships between the generated images and the corresponding astrophysical variables. This process allowed me to effectively analyze the performance and correlations of the models based on the generated data. Below I have showed this plot for one astrophysical variable

![alt text](image-1.png)

# Installation and Usage

```sh

git clone git@github.com:rishibabby/DeepLense_Diffusion_Rishi.git
cd DeepLense_Diffusion_Rishi
python -m scripts.run_roc

```

# Acknowledgment

I would like to thank my mentors, Pranath Reddy, Dr. Michael Toomey and Prof. Sergei Gleyzer, for their continued support and guidance.
