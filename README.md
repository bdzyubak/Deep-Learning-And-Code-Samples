## Deep Learning Sandbox 

Author: Bogdan Dzyubak, PhD

Email: illan7@gmail.com

Date: 09/20/2022

Repository: [tensorflow-sandbox](https://github.com/bdzyubak/tensorflow-sandbox)

## Purpose:

The purpose of this project is to explore neural networks including experimenting with state of the art
 architectures, and testing the importance of hyperparameters. To that end, I am building an easy to use 
 interface which allows training to be set up with default parameters in a few lines, or configured to 
 compare a dozen permutations of GANs with different generators/discriminators, for example. 

Please also see the more recent repo for PyTorch exploration: [torch-control](https://github.com/bdzyubak/torch-control)

![img.png](img.png)

## Installation
Install the following prerequisites:
1) Python >= 3.9 
2) If computation of GUI is desired, install CUDA, cuDNN, and Zlib per: 
    https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html, 
    https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
3) Run the setup.py code at the top of the project source tree (parallel to this README file). 

## Organization 

1) There are several projects in separate folders in the source tree, such as COVIDLungSegmentation,
  Cell-Nuclei-Segmentation, and KaggleHistopathologyDetection. These contain train* scripts that work as entry 
  points. Ideally, the train script will download their training data automatically, typically from
  Kaggle, but in some cases this has to be done manually from the link noted in the file. Trained 
  models will be   placed within the project folder. Due to size, this project does not trained models,
  just the code to generate them. In the future, an artifactory may be launched to house the latest model. 

2) The \Tutorials folder contains scripts based on examples on the web, refactored by the author. 
  These can be run directly but are not optimized to use the common initialization/preprocessing/training 
  that is one of the goals of this project.

3) The \shared_utils folder contains various functions related to OS and web interactions, image analysis,
 statistics, model setup, etc that may be useful standalone. Please feel free to use these. 

4) The \common_models folder adds additional models that are not available in tensorflow by default. These 
  were downloaded from various repositories and are not maintained by me. Authorship should be indicated in
  their source. 

## Operating Systems Notes 

1) This project was developed on Windows. It attempts to be OS agnostic - no hardcoded slashes, reliance on 
  python tools vs system tools, and os splits when system tools are used - but testing primarily happens on 
  Windows, so Linux patches will probably be needed. 

2) The project is aimed to work with both CPU only and GPU-capable setups. As long as CUDA and other 
  dependencies are installed, running on a GPU should "just work." GPU memory allocation is satic. If you 
  run into an out of memory issue, reduce batch size. 

## Bug reporting/feature requests

Feel free to raise an issue or, indeed, contribute to solving one on: https://github.com/bdzyubak/Deep-Learning-Sandbox/issues

## Testing Installation: 

TODO: add script with small epoch/data size, comparison of a few models, and full reporting. 

## Testing and Release: 

TODO: expand the range of unit-tests to cover critical functionality, and use these as a requirement to 
promotion to the beta branch. 