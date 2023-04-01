# Welcome to the  TF_flexfibreAABB repository

## This GitHub projec aims at exploiting Neural Networks to experimentally observe and measure small particles suspended in a viscous shear flow.

We are experimentally observing three-dimensional rotational dynamics of small axisymmetrical objects suspended in a viscous shear flow by two perpendicular cameras, i.e. in an Epipolar geometry, as displayed in Figure 1.
#### Figure 1: Panels (a): realistic rendering of the linear shear cell deployed in our experiments; (b): picture of the experimental setup; (c): sketch of the output given by the dual-camera video-recording system.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/setupcomplete.jpg?raw=true)

A typical result is shown in the following Videos 1 and 2 with a ring, where the axisymmetrical particle is driven by the mean shear in a periodic rotation.
#### Video 1: Top/Above view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/top.gif)
#### Video 2: Side view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/side.gif)

A precise measurement of their orientation dynamics is very important, allowing a direct comparison between theory and experiments and, ultimately, leading to a more accurate modelling of particle-dispersed turbulent flows, crucial in several industrial and environmental applications.
#### In an effort to observe particles with different shapes (ellipsoids, rings, asymmetric objects), we find it convenient to disregard a direct geometrical approach and deploy Convolutional Neural Networks (CNN) to directly estimate their orientation from the video recordings of the experiments.

The CNNs are trained over a small synthetic data set for each particle, generated in Blender rendering images of randomly oriented objects with two cameras in the same position as the real world. The data set for the considered ring is displayed in Figure 2. There is an unequivocal relation between virtual and physical particles, as the latter are 3D printed from the '.stl' file that stores the former.
#### Figure 2: training data set made by side and top renderings of a ring with inner radius 0.5 mm and outer radius 2.5 mm. The three components of the particle orientation vector in the flow (n_1), gradient (n_2) and vorticity (n_3) directions are also reported above each corresponding couple of images.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/training_dataset.png?raw=true)

A simple Computer Vision method based on the Watershed algorithm is developed in Python+OpenCV to perform the image segmentation and isolate the ring withing each video recording of the experiment. The typical processed frames are shown in Figure 3, where the dimensionless time is reported above each couple of images.
#### Figure 3: Time-evolution of the orientation of a ring suspended in a viscous shear flow
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/time_evolution.png?raw=true)

On [Google Colaboratory](https://colab.research.google.com/github/ddg93/TF_flexfibreAABB/blob/main/RegressDISK_multiview.ipynb) you can build, train and test the CNN, and eventually reconstruct the time-evolution of the particle orientation during an actual experiment with the considered ring. 


