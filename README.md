# PDE-Solver
This is a work in progress code to look into different approaches to use GANs to solve elliptic/parabolic PDEs.

# Running WAN codes
The newest and most accurate version of the codes is `pde-solver GAN 3.0.py` and is now much clearer with descriptions of all the classes and their inputs. To modify the code all the functions defined at the start and starting with `func` should be the ones of the problem described below. Further, the dimension of the problem and the number of points sampled are determined in the dictionary `setup` whilst the hyperparameters for the neural nets and the machine learning are stored in the dictionary `config`. When these are all set to match the problem all that is left to do is choose the number of iterations and to plug these into the function `train` at the bottom of the document.

At the bottom of the document is a chunck of code that is commented out. This code is there to evaluate what the loss function evaluates the loss to be if we feed the solution as the prediction for u.

# Running NCDE Codes
It is important to maintain the file structure as it is to ensure that the imports work correctly (sometimese it may not be possible to have files so it may be necessary to modify the import lines). It is important that all the imports are compatible and the version of `torch` for which the files work is `1.6.0`.

The only file that ever needs to be interacted with to set hyperparameters and run the code is `Solver`. Here the important values are:
- `intervals` (int): the number of time partitions that we will ultimately want to use in the logsignature
- `step` (int): the number of time points between the endpoints of the logsignature intervals (the paper says that this needs to be a large value)
- `batch_size` (int): this is the number of randomly chosen paths that we wish to place in the domain for each epoch of training (this needs to be divisible by 4 at the moment)
- `depth` (int): this is the truncation depth of the logsignature
- `iteration` (int): the number of epochs to train over

The other values that are specified are those that depend on the question and are labelled following the general form of our equation as shown below:

![equation](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%20u_%7Bt%7D-%5Csum_%7Bi%3D1%7D%5Ed%20%5Cpartial_i%20%28%5Csum_%7Bj%3D1%7D%5Ed%20a_%7Bij%7D%20%5Cpartial_j%20u%29%20&plus;%20%5Csum_%7Bi%3D1%7D%5Ed%20b_i%20%5Cpartial_i%20u%20&plus;%20cu%20%3D%20f%2C%20%26%20%5Ctext%20%7B%20in%20%7D%20%5COmega%20%5Ctimes%5B0%2C%20T%5D%20%5C%5C%20u%28x%2C%20t%29%3Dg%28x%2C%20t%29%2C%20%26%20%5Ctext%20%7B%20on%20%7D%20%5Cpartial%20%5COmega%20%5Ctimes%5B0%2C%20T%5D%20%5C%5C%20u%28x%2C%200%29%3Dh%28x%29%2C%20%26%20%5Ctext%20%7B%20in%20%7D%20%5COmega%20%5Cend%7Barray%7D%5Cright.)

for a d-dimensional domain.

The tuning code is commented out at the moment and has not yet been extensively tested by me so I am unsure whether it will work in all environments (note that at the moment the code specifies the use of a `cpu` for the tuning but this can easily be amended by specifying the number of `gpu` to also be used).

# Reference papers
1. [Weak Adversarial Networks for High-dimensional Partial
Differential Equations](https://arxiv.org/pdf/1907.08272.pdf)
2. [Solving high-dimensional partial differential equations using deep learning](https://www.pnas.org/content/115/34/8505)
3. [Neural CDES for Long Time Series via the log-ODE Method](https://openreview.net/references/pdf?id=UeC5mrH-_A)
