# NeuroScrub

<!-- ABOUT THE PROJECT -->
## About The Project

**NeuroScrub** is an approximate scrubbing technique to increase the retention time of emerging NVM-based neuromorphic fabric used for hardware acceleration of different classes of Neural Networks, e.g., densely connected and convolutional neural networks (CNNs). It divides the neuromorphic fabric into two regions: Scrub and Non-Scrub where all the +1 and -1 synaptic weights are mapped. Note that the mapping can be decided based on NVM technology. The training of the NNs adjusted accordingly to meet the requirements of the scrubbing scheme. This method allows retention fault-tolerant neuromorphic computing with virtually zero overhead. 

This repository gives insight into the proposed training technique of **NeuroScrub** and **NeuroScrub+** paper. Also, provides simulation tool for the retention faults. 

The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License

## Requirements
1. Pytorch
2. Python 3+
3. Numpy

## Usage: Trainig

* Run training MLP or CNN script 
