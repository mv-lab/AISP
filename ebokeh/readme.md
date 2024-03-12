
## [EBokehNet: Efficient Multi-Lens Bokeh Effect Rendering and Transformation](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Seizinger_Efficient_Multi-Lens_Bokeh_Effect_Rendering_and_Transformation_CVPRW_2023_paper.pdf)



#### Tim Seizinger, Marcos V. Conde, Manuel Kolmet, Tom E. Bishop, Radu Timofte

The official pytorch implementation of the paper Efficient Multi-Lens Bokeh Effect Rendering and Transformation (CVPR NTIRE Workshop 2023). This work is the state-of-the-art method for bokeh rendering and transformation and baseline of the NTIRE 2023 Bokeh Challenge.



>This paper introduces EBokehNet, a efficient state-of-the-art solution for Bokeh effect transformation and rendering. Our method can render Bokeh from an all-in-focus image, or transform the Bokeh of one lens to the effect of another  lens without harming the sharp foreground regions in the image Moreover we can control the shape and strength of the effect by feeding the lens properties i.e. type (Sony or Canon) and aperture, into the neural network as an additional input.  Our method is a winning solution at the NTIRE 2023 Lens-to-Lens Bokeh Effect Transformation Challenge, and state-of-the-art at the EBB Val294 benchmark.
 
 
### Installation

Clone the git and cd to the project folder:

```git clone [link]```

```cd EBokehNet```

This implementation is based on Pytorch and Pytorch Lightning. To install the required packages, 
run the following command:

```pip install -r requirements.txt```https://codalab.lisn.upsaclay.fr/competitions/10229#participate

### Dataset

The Bokeh Transformation dataset can be downloaded from the 
[NTIRE 2023 Lens-to-Lens Bokeh Effect Transformation Challenge](https://codalab.lisn.upsaclay.fr/competitions/10229#participate) 
website.
The Dataset should be placed in the ```dataset/Bokeh Transformation``` folder, 
but you can change the path in the config of each script.

The EBB bokeh rendering dataset can be downloaded from the
[AIM 2020 Rendering Realistic Bokeh Challenge](https://competitions.codalab.org/competitions/22220#participate)
website.
To generate the EBBVal294 split take the last 294 images from the training set move them to the validation set.
Furthermore, the dataloader expects both source and ground truth/target images to be in the same folder, 
with the same name, but ending in ```_src``` and ```_tgt``` respectively. (Same as the Bokeh Transformation dataset)

### Pretrained Models

The pretrained models can be downloaded from [Link to google drive] and should be put in the ```modelzoo``` folder.

### Usage

All scripts contain a separate configuration code in the beginning.

Depending on the task, the following scripts can be used:

For evaluation of the Bokeh Effect Transformation task with the regular model, run:
```python evaluate.py```

For evaluation of the Bokeh Effect Transformation task with the small model, run:
```python evaluate_s.py```

For evaluation of the Bokeh Effect Rendering task on EBBVal294, run:
```python evaluate_rendering.py```

If you only want to perform prediction without evaluation change the configuration accordingly.