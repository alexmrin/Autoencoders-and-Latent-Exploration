# Autoencoders and Latent Exploration

## Overview
I wanted to explore the latent space of a convolutional autoencoder, and see if my encoder could recognize patterns and dependencies within the 2 dimensional latent space.
My autoencoder was a little bit overkill where I used 3 layers of consisting of convolution -> batchnorm -> ReLU -> maxpool, then lastly an adaptive pooling to convert all of my channels into one scalar, ultimately compressing my inital 28x28 MNIST image into 64 neurons. I then further used a fully connnected layer to compress it down to 2 neurons, then applied a sigmoid activation instead of a ReLU to limit my range of numbers from 0 to 1.

![zero_ex1](/assets/autoencoder_1.png)

![zero_ex2](/assets/autoencoder_2.png)

As shown in the two decompressions of points close to (1, 1), we can see that nearby points tend to reconstruct similar images. Both points above reconstruct visually to the number 0, with slight distortions in shape as expected.


![zero_ex1](/assets/autoencoder_3.png)

![zero_ex2](/assets/autoencoder_4.png)

![zero_ex1](/assets/autoencoder_5.png)

In this example we can see as we traverse the latent space from near (0, 0) to (0.5, 0.5), we can start to see the shift in reconstructing from the number 1 to the number 9. This is quite fascinating, and the fact that the latent space groups vectors that have similar attributes together means that the visual features that we see are also observed by the autoencoder, and their distributions are centered and distributed quite normally throughout different parts of our 1 square unit latent space. Another thing to note is that since similar input images are grouped closer together, we can say that the *latent space preserves the topology of the input space*. 
