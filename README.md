# Welcome to the  TF_flexfibreAABB repository

## This GitHub projec aims at exploiting Neural Networks to experimentally observe and measure small particles suspended in a viscous shear flow.

We are experimentally observing three-dimensional rotational dynamics of small axisymmetrical objects suspended in a viscous shear flow by two perpendicular cameras, i.e. in an Epipolar geometry, as displayed in Figure 1.
#### Figure 1: Panels (a): realistic rendering of the linear shear cell deployed in our experiments; (b): picture of the experimental setup; (c): sketch of the output given by the dual-camera video-recording system.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/setupcomplete.jpg?raw=true)

A typical recording of an experiment is shown in the following Videos 1 and 2 with a ring, where the axisymmetrical particle is driven by the mean shear in a periodic rotation.
#### Video 1: Top/Above view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/top.gif)
#### Video 2: Side view of a ring rotating in a viscous shear flow.
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/side.gif)

A precise measurement of their orientation dynamics is very important, allowing a direct comparison between theory and experiments and, ultimately, leading to a more accurate modelling of particle-dispersed turbulent flows, crucial in several industrial and environmental applications.
#### In an effort to observe particles with different shapes (ellipsoids, rings, asymmetric objects), we find it convenient to disregard a direct geometrical approach and deploy Convolutional Neural Networks (CNN) to rapidly estimate their orientation from the video recordings of the experiments.

The CNNs are trained over a small synthetic data set for each particle, generated in Blender rendering images of randomly oriented objects with two cameras in the same position as the real world. The data set for the considered ring is displayed in Figure 2. There is an unequivocal relation between virtual and physical particles, as the latter are 3D printed from the '.stl' file that stores the former.
#### Figure 2: training data set made by side and top renderings of a ring with inner radius 0.5 mm and outer radius 2.5 mm. The three components of the particle orientation vector in the flow (n_1), gradient (n_2) and vorticity (n_3) directions are also reported above each corresponding couple of images.
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/training_dataset.png?raw=true)

A simple Computer Vision method based on the Watershed algorithm is developed in Python+OpenCV to perform the image segmentation and isolate the ring withing each video recording of the experiment, as visualized in the Video 3. 
#### This is very important to detect the particles, separating them from small bubbles, dust, suspended within the flow as well as, for the side view, to see through the glass wall and the transparent plastic belt.
```python
### 1: Original grayscale image. Many small bubbles are trapped in the viscous fluid around the object; 
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
### 2: First estimate of the separation between the object and the background by thresholding binarization (Otsu's method): colour scale from yellow (high value, object) to purple (low value, background);
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
### 3: Noise removal by Dilation: the object is dilated to find the sure background, while the small bubbles are cancelled;
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel,1)           
sure_bg = cv2.dilate(opening,kernel,iterations=1)
### 4: Distance transformation: each pixel is labelled according to its Euclidean distance from the closest zero (purple, background) pixel;
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3) ###size of mask
### 5: Identification of the sure object by thresholding (70 %) on the distance transformation image;
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
### 6: Identification of the unknown region by subtracting the sure object from the sure background;
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
### 7: Preliminary labelling of the sure background (green), sure object (yellow) and the unknown region (purple) by a connected components labelling algorithm; 
ret, markers = cv2.connectedComponents(sure_fg) # Add one to all labels so that sure background is not 0, but 1
### 8: Final labelling obtained using the Watershed algorithm;
markers = markers +1 # Now, mark the region of unknown with zero
markers[unknown==255] = 0                
res = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
markers = cv2.watershed(res,markers)
res[markers == -1] = [255] ###dark contours
```
#### Video 3: steps of the implemented Watershed algorithm (see the file description for the details).
![](https://github.com/ddg93/TF_flexfibreAABB/blob/main/watershed.gif)

The typical processed frames of one experiment are shown in Figure 3, where the dimensionless time is reported above each couple of images.
#### Figure 3: Time-evolution of the orientation of a ring suspended in a viscous shear flow
![alt text](https://github.com/ddg93/TF_flexfibreAABB/blob/main/time_evolution.png?raw=true)

#### At this point, we introduce the CNN. 
Having already post-processed the video recordings with the Python+OpenCV implementation of the Watershed method, the Neural Network has a simple architecture, taking the two perpendicular frames as input and returning the three components of the particle orientation vector as output. 
Initially, we work with a single Convolutional layer for image, followed by a Pooling layer. Then, the tensors are concatenate before flattening them. Finally, one Dense layer regresses the three components of the particle orientation vector.
```python
###Input shape
input_shape = (100, 100, 3)
###Define the first input for image 1
input1 = Input(shape=input_shape)
###Convolutional layers for image 1
conv1_1 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(input1)
maxpool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
###Define the second input for image 2
input2 = Input(shape=input_shape)
###Convolutional layers for image 2
conv2_1 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(input2)
maxpool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
###Concatenate the output of the convolutional layers for both images
concat = concatenate([maxpool1_1, maxpool2_1])
###Flatten the output of the convolutional layers
flatten = Flatten()(concat)
dense1 = Dense(64, activation='relu')(flatten)
output = Dense(3, activation='linear')(dense1)
###Define the model with two inputs and one output
model = Model(inputs=[input1, input2], outputs=output)
```

The model is compiled with a custom loss function, which calculates the Euclidean norm of the difference between the True and Predicted values of the particle orientation vector.
```python
def custom_loss(y_true,y_pred):
    norm = tf.norm((y_true-y_pred),axis=-1)
    loss = tf.math.reduce_sum(norm)
    return loss
```

## Build, train and test the CCN on  [Google Colaboratory](https://colab.research.google.com/github/ddg93/TF_flexfibreAABB/blob/main/RegressDISK_multiview.ipynb) in order to measure the time-evolution of the particle orientation during the experiment with the considered ring. 





