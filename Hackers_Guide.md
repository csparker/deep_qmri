Welcome Hackers! 

Below is a more detailed [Background](#rationalle) for the project, FYI. 

The [Set-up](#set-up) section lists some things you will need to do before you start hacking. 

Once you're all set up, the [Hacking Tasks](#hacking-tasks) and [Resources](#resources) sections are there to help us complete the project!

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

A big problem with current unsupervised machine learning approaches is that the cost function is wrong. This is because current methods most typically use sum of squared differences as the cost function, which only works when the MRI noise is gaussian distributed. Effectively, it is always assuming the wrong model for the data, because in fact, the noise in MRI images is rician distributed. 

Lack of gaussianity in MRI images is especially apparent for low SNR MRI images - at low SNR the Rician distribution deviates substantially from a gaussian distribution. The result is that the signal predictions are optimised incorrectly, resulting in incorrect parameter estimates. 

This has not been previously addressed because the Rician probability density function is not differentiable, and differentiable loss functions are needed for backpropagation to work in machine learning. Recently, a paper has introduced a [differentiable approximation of the Rician distribution](https://link.springer.com/chapter/10.1007/978-3-031-11203-4_16) and a differentiable log-likelihood for it. The rician noise model can therefore be adopted and incorporated into unsupervised machine learning to make parameter estimates more accurate.

<br>

# Set-up

First, you will need to install [Python](https://www.python.org/downloads/).

You will then need to install some Python packages. You can use the "pip3" command, like this:
```
pip3 install some-package
```

The following packages are needed:
- [numpy](https://numpy.org/install/)
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [PyTorch](https://pytorch.org/TensorRT/tutorials/installation.html)
- [tqdm](https://pypi.org/project/tqdm/)
- [Jupyter Notebook](https://jupyter.org/install) (optional)


That's it, you're all set up. Now let's start Hacking!

<br>

# Hacking Tasks


Below is a list of tasks for Hackathon, with hints/suggestions under each.

This is only meant as a rough guide. Feel free to suggest or discuss other ideas that might not be on the list!

<br/>


<!-- 
<details>
<summary>How do I dropdown?</summary>
<br>
This is how you dropdown.
</details> -->



<details>
<summary><h3>1. Incorporate the Rician likelihood loss function into unsupervised learning</h3></summary>
<br>

- Compare the Rician distribution with its [differentiable approximation](https://link.springer.com/chapter/10.1007/978-3-031-11203-4_16).
	- what value of Nk leads to a good approximation?

- Using simulated Rician data, compare the likelihood of the data under a Rician distribution and under the differentiable approximation of the Rician distribution.
	- Are the likelihoods highly correlated?

- Create a custom PyTorch loss function that calculates the log-likelihood of DWI data under the approximate Rician distribution.
	- Use a fixed value of sigma for now.

- Add the new PyTorch loss function into the unsupervised learning network.
	- You can use the logsumexp() function described in Simpson et al (2021), which is supposed to be numerically stable.

- Use the network to estimate IVIM parameters from simulated data.
	- Do the parameter estimates look reasonable?

- Allow the network to learn the sigma value.
	- We will need to change the network architecture.
</details>

<br/>

<details>
<summary><h3>2. Evaluate accuracy of parameter estimation</h3></summary>
<br>

- Quantify bias and variance when using the sum of squared loss function.
	- Assess for a single ground truth parameter value.

- Quantify bias and variance when using the Rician log-likelihood loss function.
	- Is bias significantly reduced? What about variance?

</details>


<br/>

<details>
<summary><h3>3. Maximum aposteriori inference</h3></summary>
<br>

- Specify plausible priors on each parameter based on the literature.
	- These may be specific to a particular anatomical region.

- Adapt the network to perform maximum aposteriori inference

</details>

<br/>
<details>
<summary><h3>4. Test on Real data</h3></summary>
<br>

- Download some real DWI data (i.e. from the Human Connectome Project)
	- I can provide you with some data.

- Re-train the network on synthetic DWI data acquired with the same settings (b-values) as the real data.
	- You could create a seperate tutorial for the real data application.

- Estimate the parameter maps for real data
	- Save the parameter maps as nifti files! This will be useful later.

- Repeat the above but for the sum of squares loss function
	- Can the network settings be saved to prevent re-training?

- Compare the parameter estimates between sum of squares and Rician likelihood loss function
	- Variance of parameters within ROIs might be a good evaluation metric (lower is probably better)

</details>
<br/>

<details>
<summary><h3>5. Try another DWI model</h3></summary>
<br>


- Specify the forward model and code this into the network.
	- The diffusion kurtosis model (see Resource section!) is a good choice because it is even more reliant on low SNR images.

- Re-train the network using simulated data and evaluate parameter estimation performance
	- Make sure the acquisition settings are suitable for the model.
	- Use an appropriate range of parameter values.

- Assess parameter estimation performance on real data
	- Is the benefit greater for this DWI model than the IVIM model?

</details>
<br/>

<details>
<summary><h3>6. Generalise the Code</h3></summary>
<br>

- Different DWI models
- Inclusion/exclusion of priors
- The loss function

</details>

<br/>

<details>
<summary><h3>7. Alternatives to the Rician likelihood</h3></summary>
<br>

- Offset gaussian noise model
	- This uses sum of squares loss function but to an offset predicted signal
	- see the Resources section for a link to the paper on Gaussian offset noise model

</details>

<br/>


**Project Outcomes**

- Main aim:
	- Create a tutorial on using the Rician likelihood approach to unsupervised qMRI (simulated data).

- If time, we could also make tutorials/walk-through on:
	- Rician likelihood approach applied to real data
	- Maximum aposteriori inference unsupervised qMRI (simulated data)
	- Maximum aposteriori approach applied to real data

- Theoretical description of the methods

<br>

# Resources

*Hackathon-related*

CMIC Hackathon website\
https://cmic-ucl.github.io/CMICHACKS/

Project Issues\
https://github.com/CMIC-UCL/CMICHACKS/issues/7

<br/>

*Repository internal links*

Main page\
https://github.com/csparker/deep_qmri

Hackers Guide\
https://github.com/csparker/deep_qmri/blob/master/Hackers_Guide.md

Our tutorial (in progress)\
https://github.com/csparker/deep_qmri/blob/master/deep_qmri_rician.ipynb

A useful template for unsupservised qMRI tutorial (Jupyter notebook)\
https://github.com/csparker/deep_qmri/blob/master/deep_qmri_leastsquares_demo.ipynb

A useful template for unsupervised ML qMRI tutorial (Python code)\
https://github.com/csparker/deep_qmri/blob/master/deep_qmri_leastsquares_demo.py

<br/>


*Relevant Publications*

Paper on differentiable Rician log-likelihood\
https://link.springer.com/chapter/10.1007/978-3-031-11203-4_16

Deep learning of qMRI paper (base network on which we're building)\
https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.27910

A more recent deep learning of qMRI paper (good for evaluation methods)\
https://arxiv.org/abs/2205.05587

Another paper on Deep learning of qMRI parameters and training data distribution\
https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29014

IVIM diffusion MRI model\
https://pubs.rsna.org/doi/10.1148/radiology.161.2.3763909

Diffusion kurtosis model\
https://www.ajronline.org/doi/full/10.2214/AJR.13.11365

Offset gaussian noise model\
https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.25080

<br/>

*Other*

Repository for the original deep learning of qMRI (this is where I got the template)\
https://github.com/sebbarb/deep_ivim




