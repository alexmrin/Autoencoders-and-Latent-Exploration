# Autoencoders and Latent Exploration

## Overview
I wanted to explore the latent space of a convolutional autoencoder, and see if my encoder could recognize patterns and dependencies within the 2 dimensional latent space.
My autoencoder was a little bit overkill where I used 3 layers of consisting of convolution -> batchnorm -> ReLU -> maxpool, then lastly an adaptive pooling to convert all of my channels into one scalar, ultimately compressing my inital 28x28 MNIST image into 64 neurons. I then further used a fully connnected layer to compress it down to 2 neurons, then applied a sigmoid activation instead of a ReLU to limit my range of numbers from 0 to 1.


