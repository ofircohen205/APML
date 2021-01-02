# Exercise 5 - Generative Latent Optimization

In order to execute the program you need to pass the following arguments:
1. name - the name of this run
2. log_dir - directory for tensorboard logs (common to many runs)

Optional arguments:
1. epochs - default is 50
2. batch_size - default is 128
3. code_dim - default is 128
4. lr - learning rate, default is 1e-3
5. stddev - sigma of normal distribution, default is 0.3

About:
This program implements the Generative Latent Optimization (GLO) algorithm on MNIST data-set, means that given the above hyper-parameters, our model returns a generated set of images of digits that are similar to the ground truth (the original data-set images).