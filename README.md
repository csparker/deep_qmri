# Deep Learning of qMRI Parameters using Rician Likelihood-based Loss Function

This repository contains resources for performing accurate quantitative MRI parameter estimation using unsupervised deep learning:
- A short description of the problem and how we addressed it.
- Python notebook describing how to implement the technique in PyTorch.
- Python code for the PyTorch implementation.
- Implementations for Keras and TensorFlow.

In a nutshell, the project aims to make deep learning of quantitative parameters from MRI more accurate. We aim to acheive this by correctly handling the distribution of MRI image noise when evaluating the loss function. 

By using a Rician Likelihood-based loss function, parameter estimation bias is theoretically and empirically removed, compared to using the Mean Squared Error (MSE) loss function.

Our publication describes the approach and evaluates parameter estimation performance:

Parker, CS., Schroder, A., Epstein, S., Cole, J., Zhang, G. (2023) [Rician Likelihood-based loss function for unsupervised learning of quantitative MRI parameters](https://onlinelibrary.wiley.com/action/doSearch?AllField=technical+note&SeriesKey=15222594). *Journal*. 

(Author list and DOI T.B.C) 









