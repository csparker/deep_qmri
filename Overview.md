Below is a more detailed [Background](#rationalle) for the project. The [Experiments](#experiments) section provides details how we tested the approach. The [Findings](#findings) section shows the main results which empirically demonstrate the advantage of using our proposed method. Finally, the [Summary](#summary) section provides an overview of the findings and conclusions.


# Background

<br>

**What is quantitative MRI?**

Quantitative MRI derives countable explanations (i.e. quantities) from the data we measure in an MRI image. This is acheived by first constructing an explanation for the data. The explanation, or model, provides a mapping from the quantities we care about (e.g. number of cells, fiber orientation) to the MRI data we measured. Then, we find out how much of each quantity in the model there should be so that the MRI data predicted by the model matches the data we observed. This is called 'model fitting' or 'parameter estimation'.

<br>

**Why do we need machine learning?**

Traditionally model fitting is performed on each voxel, one after the other (in qMRI each voxel contains multiple data points). However, our explanations have got more and more detailed, and so our models have become more and more complicated. The process of parameter estimation has become increasingly difficult due to the shear numbers of parameters that need to be tuned and the possibility that an incorrect fitting may appear to be the best fit when it is not. 

To overcome this, machine learning has been used to construct a direct mapping from the data we observe to the parameter estimates, taking advantage of the ability of neural networks to represent a vast range of highly complex functions. Unsupervised learning is particularly desireable because it does not require us to have any ground truth labels on the parameters, which in practice is never possible. 

In comparison to voxel-wise fitting, unsupervised learning tries to match the predicted signal with the observed data. The parameters are derived from an encoded latent space. In other words, the network learns to map from the data to the best predicted data via the parameters of interest, by using the model predictions, similarly to an autoencoder. 

<br>

**What research question is our project addressing?**

A big problem with current unsupervised machine learning approaches is that the cost function is wrong. This is because current methods most typically use sum of squared differences as the cost function, which only works when the MRI noise is gaussian distributed. Effectively, it is always assuming the wrong model for the data, because in fact, the noise in MRI images is [rician distributed](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.1910340618?casa_token=d-tl0knH3_oAAAAA:jWXOQSUz9xkZ4KyDXchZM7SlpVg2-hzx3VoZEM5sF2zXkP2NrZ0vhBy3MHfhKe35suxt72nO75gMsE9Z). 

Lack of gaussianity in MRI images is especially apparent for low SNR MRI images - at low SNR the Rician distribution deviates substantially from a gaussian distribution. The result is that the signal predictions are optimised incorrectly, resulting in incorrect parameter estimates. 

This has not been previously addressed because implementing the Rician Likelihood based loss functions poses significant challenges. Firstly, the Rician probability density function is non-computable with certain inputs. Previous work has addressed this by introducing an approximation of the Rician distribution. However, our experiments show this approximation is innacurate at high SNR, biasing parameter estimates. Therefore, we employ a differentiable version of the Rician distribution which is computable for all inputs. Furthermore, we improve the speed of its computation using vectorisation, making it easily applicable to large training sets.


Our experiments showed that by incorporating the Rician noise model into unsupervised machine learning we are able to reduce bias in parameter estimation, making quantitative MRI via unsupervised learning more accurate.

<br>

# Experiments


<br>

# Findings

# Summary


<br/>

