# Implementing OCR from scratch using K Nearest Neighbours

We get the training set, i.e., the images and labels of the numbers we want to identify from the MNIST database

[MNIST Database](http://yann.lecun.com/exdb/mnist/)

The files in this link often prove troublesome to download, so use Wayback Machine to access an earlier version of the webpage and download the four files from there

We will be implementing k nearest neighbors. The difference between this algorithm and usual machine learning algorithms is that most machine learning algorithms have a training stage, they learn the separating line and then test it. On the other hand, k nearest neighbors does not have this training step in the beginning, it is a lazy learning algorithm. It trains as we try to figure out the new input.

The data are the images themselves, and the labels are the true values of the digits themselves

<img width="368" alt="Training Set Image File" src="https://github.com/frindle18/OCR-From-Scratch/assets/170041321/a2bd9f58-30ca-4232-8476-fe2cad5088b5">

The magic number indicates the type of file

Convention:
X means dataset, y means labels

<img width="373" alt="Training Set Label File" src="https://github.com/frindle18/OCR-From-Scratch/assets/170041321/afa71e59-552e-4a0d-a015-60911790b3cd">


- We need the distance between two points

We calculate the normal Euclidian distance in n dimensional space, where n is the number of features. For this, we need a vector consisting of the features, here, the pixels. But the way we've read the file is to make a list of list of these feature vectors. What we want now is just a list of all pixels.

For a particular point (the testing point), k nearest neighbors looks at every other point and calculates the distance, then it takes the point with smallest distance.

k is called the 'hyper parameter', which means it's a number that we can change as per our wish to affect the algorithm such that we can make the output of the algorithm more precise.

Credit: [clumsy computer](https://youtu.be/vzabeKdW9tE?si=6MIlibIbc2qV7Zjj)
