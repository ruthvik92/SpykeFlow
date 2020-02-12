# Introduction
* This is a tool for unsupervised feature extraction with spiking neural networks.
* Neurons are non-leaky integrate and fire. There are methods to enforce lateral inhibition, STDP competition.
* It provides insights like spike activity, feature extraction, animation of synapse evoltuion for each layer etc. 
* It also provides feature classification class and a few other jupyter notebooks with misc codes.
# Organization
* `AllDataSets` folder should contain the data that you wish to work with.
* `spikeflow` folder contains all the important classes of **SpikeFlow**.
* `notebooks` contains miscelleneous jupyter notebooks for classification, plots, etc.
* `outputs` contains all the outputs/plots generated so far with **SpikeFlow**.
* `notebooks/main_emnist.ipynb` shows an example to train a spiking convolutional layer with two conv and pool layers
with **EMNIST** dataset and effects of lateral inhibition on spike activity and feature extraction inside the network.
* `notebooks/main_facebike.ipynb` shows an example to train a spiking convolutional layer with three conv and pool layers
with **facebikes** dataset and effects of lateral inhibition on spike activity and feature extraction inside the network.
* Exhaustive list of requirements in listed in `requirements.txt` however most important requirements are
* `NumPy==1.15.3`, `SciPy==1.1.0`, `Progressbar==2.3`, `Theano==0.8.0`, `Pandas==u'0.20.3`, `Pickle==$Revision: 72223$`,
* `Matplotlib==2.2.3`, `Keras==2.2.4`, `Sklearn==0.19.1`, `OpenCv==3.3.0`.
* Unfortunately, the code is in `Python2.7`. I will work on porting it to `Python3.7` soon.
# Publications
* 

