# Project Title and Description

Title : Diffusion Models for Gravitational Lensing Simulation

Description : Strong gravitational lensing is a promising probe of the substructure of dark matter to better understand its underlying nature. Deep learning methods have the potential to accurately identify images containing substructure and differentiate WIMP particle dark matter from other well-motivated models, including axions and axion-like particles, warm dark matter, etc.

Traditional simulations of gravitational lensing are time-consuming and require extensive computational resources. This project proposes the use of diffusion models, a class of generative models known for their ability to produce high-quality, detailed images from a distribution of noise, to simulate strong gravitational lensing images. We aim to generate realistic simulations of gravitational lensing events that can be used to augment datasets for machine learning models and facilitate the development of better domain adaptation and self-supervised models aimed at bridging the gap between simulated and real images of gravitational lensing. Furthermore, we will also investigate leveraging conditional diffusion models to generate gravitational lensing simulations by conditioning the model on specific parameters related to the lensing events, such as the mass distribution of the lensing galaxy, orientation, and the redshift of both the source and the lens.

# Class Conditional Diffusion model

I have trained a diffusion model using labels as a condition, with the specific labels being "axion," "cdm," and "no_sub." In addition, I have trained a ResNet model on these same classes. To evaluate the performance, I generated images using the trained diffusion model and then used these images to plot a Receiver Operating Characteristic (ROC) curve. This allowed me to assess the effectiveness of the ResNet model in distinguishing between the different classes based on the generated images.

![alt text](image.png)

# Variable Conditional Diffusion model

I have trained a diffusion model conditioned on all astrophysical variables, alongside training a ResNet-18 model using these variables for regression. Following the training, I generated images using the diffusion model and created scatter plots to visualize the relationships between the generated images and the corresponding astrophysical variables. This process allowed me to effectively analyze the performance and correlations of the models based on the generated data. Below I have showed this plot for one astrophysical variable

![alt text](image-1.png)

# Requirements

pip install -r requirements.txt

# Installation and Usage

```sh

git clone git@github.com:rishibabby/DeepLense_Diffusion_Rishi.git
cd DeepLense_Diffusion_Rishi
python -m scripts.run_roc

```

# Acknowledgment

I would like to thank my mentors, Pranath Reddy, Dr. Michael Toomey and Prof. Sergei Gleyzer, for their continued support and guidance.
