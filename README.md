# Welcome to the  TF_flexfibreAABB repository

## This GitHub projec aims at building a Neural Network that reconstructs the three-dimensional conformation of a flexible fibre in an epipolar geometry. 
Motivation for this effort comes from its application in the field of experimental fluid mechanics, as we are currently building a simple experiment to observe the dynamics of flexible particles suspended in viscous shear flows (see [our previous project with rigid ellipsoids and cylinders](https://github.com/ddg93/JOposeAABB) for more details and a geometrical approach to the regression of their three-dimensional orientation).

## Roadmap:
### 1) Development of a Neural Network that detects flexible fibres 
At first, we focus on developing a simple Neural Network to measure the Axes-Aligned Bounding Box (i.e. the minimum enclosing rectangle that is aligned with the image edges) containing the fibre for a single view. To do so, we train and test on virtual images, generated using Blender.

Build, train and test the model on [Google Colaboratory](https://colab.research.google.com/github/ddg93/TF_flexfibreAABB/blob/main/Fibre_AABB_detection.ipynb).
