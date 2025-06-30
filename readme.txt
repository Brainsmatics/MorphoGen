# MorphoGen

## Overview
Here we introduce the environment dependencies, datasets used, file description, and the code execution.

## Dependencies
python==3.8
pytorch==2.0.0
torchvision==0.15.1
cudatoolkit==12.2

## Dataset
Neuron data is sourced from the study:  
Gao, Le, et al. "Single-neuron projectome of mouse prefrontal cortex." Nature neuroscience 25.4 (2022): 515-529. 

- **CT subtypes (45-52)**: 1,085 neurons (all subtypes)  
- **PT subtypes (57-64)**: 1,005 neurons  
- **IT subtypes (34-44)**: 985 neurons  

## File Description
- `sub_process.py`: Converts raw SWC files to standardized point cloud data.
- `distort.py`: Distorts true branches to learn the mapping back to original state.
- `DDPM_train.py`: Trains the denoising diffusion probabilistic model to predict global structures.
- `Auxiliary_train.py`: Trains the auxiliary CNN networks to optimize the local structures.
- `morphology_gen.py`: Generates new morphology point clouds and converts into SWC files.

## Code Execution
train the DDPM：
```
python DDPM_train.py --dataroot ${dataroot} --model_dir${model_dir} --device ${device}
```
train the Auxiliary CNN：
```
python Auxiliary_train.py
```
generate new neuron morphology:
```
python morphology_gen.py --dataroot ${dataroot} --model${model} --device ${device} --generate_dir ${generate_dir}
```
