Welcome Hackers! 

Below is a more detailed [Rationalle](#rationalle) for the project. In the [Set-up](#set-up) section there are some instructions on things you will need to do before you start hacking. 

Once you're all set up, the [Hacking Tasks](#hacking-tasks) and [Resources](#resources) sections are there to help us complete the project!

# Rationalle

Quantitative MRI aims to derive countable explanations (i.e. quantities) from the data we measure in an MRI image. This is acheived by first constructing an explanation for the data. The explanation, or model, maps from the quantities we care about (e.g. number of cells, fiber orientation) to the data. Then, we find out how much of each quantity in the model there should be in order that the data predicted by the model matches the data we observed. This is called 'model fitting' or 'parameter estimation'. 	

Traditionally model fitting is performed on each voxel, one after the other (in qMRI each voxel contains multiple data points). However, as our explanations have gotten more and more detailed, and hence our models more and more complicated, the process of parameter estimation has become increasingly difficult due to the shear numbers of parameters that need to be tuned and the possibility that an incorrect fitting may appear to be the best fit when it is not. 

To overcome this, machine learning has been used to construct a direct mapping from the data we observe to the parameter estimates, taking advantage of the ability of neural networks to represent a vast range of highly complex functions. In particular, an unsupervised learning approach is desireable because it does not require us to have any ground truth labels on the parameters, which in practice is never possible. Instead, unsupervised learning tries to match the predicted signal with the observed data where the parameters are deriveable from an encoded latent space. In other words, the network learns to map from the data to the predicted data via the parameters of interest and the specified forward model, similarly to an autoencoder. 

However, a big problem with current unsupervised approaches is that the predicted signal is always wrong, by design. This is because current methods most typically use sum of squared differences as the cost function, which only works when the MRI noise is gaussian distributed. Effectively, it is always assuming the wrong model for the data, because in fact, the noise in MRI images is rician distributed. This is especially important for parameters derived from low SNR MRI images, because at low SNR the Rician distribution deviates substantially from a gaussian distribution. The result is that the signal predictions are optimised incorrectly, resulting in incorrect parameter estimates. This problem has not previously been addressed because the Rician distribution is not differentiable, and differentiable loss functions are needed for backpropagation to work in machine learning. Recently, a paper has introduced a differentiable approximation of the Rician distribution and a differentiable log-likelihood for it. The rician noise model can therefore be adopted and incorporated into unsupervised machine learning to make parameter estimates more accurate.

# Set-up

First, you will need to install [Python](https://www.python.org/downloads/).

You then need to install some Python packages which have the functionality we need. 

To install Python packages you can use the "pip3" command, like this:
```
pip3 install some-package
```

The following packages are needed:
- [numpy](https://numpy.org/install/)
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [PyTorch](https://pytorch.org/TensorRT/tutorials/installation.html)
- [tqdm](https://pypi.org/project/tqdm/)

That's it, you're all set up. Now let's start Hacking!


# Hacking Tasks

# Resources

CMIC Hackathon website
https://cmic-ucl.github.io/CMICHACKS/

Project Issues
https://github.com/CMIC-UCL/CMICHACKS/issues/7

Original deep learning of qMRI paper
https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.27910

Deep learning of qMRI parameters and training data distribution
https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29014

Most recent deep learning of qMRI paper 
https://arxiv.org/abs/2205.05587

Paper on differentiable Rician log-likelihood
https://link.springer.com/chapter/10.1007/978-3-031-11203-4_16


