### Before Start

The model is trained with [PyTorch](https://pytorch.org) and the codes for MCRG analysis  are written in C++. Please install PyTorch  and make sure the compiler supports C++11. The figures are plotted with [Matplotlib](https://matplotlib.org). Please install it as well.

The code by default run on GPU, but fallback to CPU if no GPU is available.

### Examples
First, we prepare a dataset with 10,000 spin configurations MC sampled 32x32 Ising model at the critical temperature.

```
python generate.py --input-width 32 --n-train 10000
```

The program will create a `datasets` folder (in the current directory if not otherwise specified) to store datasets.

Second, we train a weight factor on the dataset.

```
python example.py --input-width 32
```

The program will create a `plots` folder to store filter images and model parameters later used in MCRG analysis.

There are other arguments relating to training the model.
```
optional arguments:
  --epochs N         number of epochs to train (default: 100)
  --batch-size N     input batch size for training (default: 50)
  --n-gibbs-steps N  number of Gibbs updates per weights update (default: 3)
  --filter-width N   linear dimension of filter (default: 8)
  --learning-rate F  learning rate of ADAM (default: 0.001)

```

Third, do MCRG analysis.

```
python analysis.py --ncpus 10
```
The program will perform 10 independent analyses in parallel each with 10,000 newly generated samples. See `plots/res` for the results (by default gives the thermal exponent for the weight factor at the last epoch). Note, however, the statistics reported are intrinsic to the specific filter trained at the second step and to the specific data generated at the first step. 



