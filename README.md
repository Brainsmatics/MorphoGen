# MorphoGen
Efficient Unconditional Generation of Long-Range Projection Neuronal Morphology via a Global-to-Local Framework. 
Code is almost one-click runnable.
Below we introduce the environment dependencies, file description, datasets used, and the code execution. Paper is [here](www.baidu.com). 

## Dependencies
python==3.8.5, pytorch==1.8.2, torchvision==0.9.2, cudatoolkit==11.1

See `requirements.txt` for detailed environment specifications.

## File Description
- `sub_process.py`: Converts raw SWC files to standardized point cloud data.
- `distort.py`: Distorts true branches to learn the mapping back to original state.
- `DDPM_train.py`: Trains the denoising diffusion probabilistic model to predict global structures.
- `Auxiliary_train.py`: Trains the auxiliary CNN networks to optimize the local structures.
- `morphology_gen.py`: Generates new morphology point clouds and converts into SWC files.

## Dataset
Long-range neuronal data is sourced from [this study](https://www.nature.com/articles/s41593-022-01041-5).  

- **CT subtypes (45-52)**: 1,085 neurons (all subtypes)  
- **PT subtypes (57-64)**: 1,005 neurons  
- **IT subtypes (34-44)**: 985 neurons  

## Code Execution
train the [DDPM](https://github.com/DiT-3D/DiT-3D)：
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

## Citation
If you find this repository useful, please cite our paper:

## Acknowledgement
Thanks for the wonderful work [DiT-3D](https://github.com/DiT-3D/DiT-3D).
