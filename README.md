# Welcome to the  TF_flexfibreAABB repository

## This GitHub projec aims at building a Neural Network that reconstructs the three-dimensional conformation of a flexible fibre in an epipolar geometry. 

The experimental observation of the dynamics of flexible objects suspended in flows implies the deployment of multiple cameras in order to reconstruct their three-dimensional conformation. The most simple scenario is that of a single particle in an epipolar geometry, as rendered in Figure 1.

#### Figure 1: Representation of the epipolar geometry created in Blender to prepare the virtual images of the flexible fibres
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/epipolar_flex.jpg?raw=true)

Motivation for this effort comes from its application in the field of experimental fluid mechanics, as we are currently building a simple experiment to observe the dynamics of flexible particles (fibres and disks) suspended in viscous shear flows (see [our previous project with rigid ellipsoids and cylinders](https://github.com/ddg93/JOposeAABB) for more details and a geometrical approach to the regression of their three-dimensional orientation).

## Roadmap:
### 1) Development of a Neural Network that detects flexible fibres 
At first, we focus on developing a simple Neural Network that will take as input a single view of the fibre and will regress its Axes-Aligned Bounding Box (i.e. the minimum enclosing rectangle that is aligned with the image edges). We train and test on virtual images, generated using Blender, some of them displayed in the panels of Figure 2. True values of the AABBs are calculated detecting the fibres with the OpenCV implementation of the [Canny method](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), plotted in cyan in the panels of Figure 2.

#### Figure 2: Visualization of the Training data set, made of virtual images of flexible fibres with their AABB highlighted in cyan.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/AABB_flex_true.jpg?raw=true)

Build, train and test the model on [Google Colaboratory](https://colab.research.google.com/github/ddg93/TF_flexfibreAABB/blob/main/Fibre_AABB_detection.ipynb).


### 2) To be continued...
