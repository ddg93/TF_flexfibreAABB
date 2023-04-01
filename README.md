# Welcome to the  TF_flexfibreAABB repository

## This GitHub projec aims at exploiting Neural Networks to experimentally observe and measure small particles suspended in a viscous shear flow.

As detailed in [our previous work](https://github.com/ddg93/JOposeAABB), we are observing three-dimensional rotational dynamics of small axisymmetrical objects suspended in a viscous shear flow by two perpendicular cameras, i.e. in an Epipolar geometry, displayed in Figure 1.

#### Figure 1: Panels (a): realistic rendering of the linear shear cell deployed in our experiments; (b): picture of the experimental setup; (c): sketch of the output given by the dual-camera video-recording system.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/setupcomplete.jpg?raw=true)

A typical result is shown in the following Videos 1 and 2 with a ring, where the axisymmetrical particle is driven by the mean shear in a periodic rotation.
#### Video 1: Top/Above view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/top.gif)
#### Video 2: Side view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/side.gif)

A precise measurement of their orientation dynamics is very important, allowing a direct comparison between theory and experiments and, ultimately, leading to a more accurate modelling of particle-dispersed turbulent flows. 

#### Figure 2: Representation of the epipolar geometry created in Blender to prepare the virtual images of the rings.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/blender_setup.jpg?raw=true)


In an effort to observe particles with different shapes such as rings and even fore-aft asymmetric objects, we find it convenient to deploy Convolutional Neural Networks to directly estimate their orientation from the video recordings of the experiments.

The CNNs are trained over a synthetic data set, generated in Blender rendering images of randomly oriented particles in an epipolar geometry, displayed in Figure 2.

#### Figure 2: training data set made by side and top renderings of a ring with inner radius 0.5 mm and outer radius 2.5 mm. The three components of the particle orientation vector in the flow (n_1), gradient (n_2) and vorticity (n_3) directions are also reported above each corresponding couple of images ([details about the experimental set-up can be found in our previous work](https://github.com/ddg93/JOposeAABB)).
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/training_dataset.png?raw=true)

Having 3D printed the ring whose '.stl' file was imported in Blender to generate the synthetic data set, we use this particle to perform one experiment. Then, a simple Computer Vision method based on the Watershed algorithm is developed in Python+OpenCV to perform the image segmentation and isolate the ring withing each video recording of the experiment. The typical processed frames are shown in Figure 3, where the dimensionless time is reported above each couple of images.

#### Figure 3: Time-evolution of the orientation of a ring suspended in a viscous shear flow
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/time_evolution.png?raw=true)

On [Google Colaboratory](https://colab.research.google.com/github/ddg93/TF_flexfibreAABB/blob/main/RegressDISK_multiview.ipynb) you can build, train and test the CNN, and eventually reconstruct the time-evolution of the particle orientation during an actual experiment with the considered ring. 


