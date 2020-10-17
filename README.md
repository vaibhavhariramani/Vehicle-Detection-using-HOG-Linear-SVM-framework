# HOG-based linear SVM vehicle detection

The files in this repo form a general framework for training and utilizing a HOG-based linear SVM to detect vehicles (or any other object) in a video. **Although I'm not enrolled in the course and have never used Udacity, this project was inspired by the [vehicle detection project](https://github.com/udacity/CarND-Vehicle-Detection) from Udacity's self-driving car nanodegree program**. 
**I've used Udacity's sample image datasets (which were themselves taken from the [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) datasets) to train the SVM, and the project video provided by Udacity to test my SVM and object detection pipeline.**

![final_video_screencap](https://github.com/vaibhavhariaramani/hog--svm-vehicle-detector-and-Object-Detection/blob/master/images/final_bounding_boxes.png)

[Click here to watch the final video](https://youtu.be/rV89KM6izi8) with bounding boxes drawn around detected cars.

**﻿Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.**

## Project overview
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Extract features from labeled (positive and negative) sample data, split into training, cross-validation, and test sets. Train classifier.
* For feature extraction, convert images to the desired color space, select the desired channels, then extract HOG, color histogram, and/or spatial features.
* Detect and draw bounding boxes around objects in a video using a sliding window and smoothed heatmap.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Note: (Code snippets taken from Udacity SD course)

[//]: # (Image References)
[image1]: ./results/output_images/car_not_car.png
[image2]: ./results/output_images/HOG_example.png
[image3]: ./results/output_images/sliding_windows.png
[image5]: ./results/output_images/bboxes_and_heat.png
[image7]: ./results/output_images/output_bboxes.png
[image8]: ./results/output_images/video_frame.png
[image9]: ./results/output_images/colorspace_yuv.png
[image10]: ./results/output_images/not_car.png
[video1]: ./results/output_videos/project_video.mp4


### README

### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images.

The code for this step is contained in the first 12 code cells of the IPython notebook (vehicle-detection.ipnyb). 

I started by reading in all the [`vehicle`](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [`non-vehicle`](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images.  Here is an example of one of each of the [`vehicle`](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [`non-vehicle`](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) classes:

![alt text][image1]
![alt text][image10]

Then using different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters using the grid search and found that HOG channels encode most of the required features. The params gave a little more than 99% accuracy with the test. set. 
Hist bins and color spacial bins are not that useful features. Increasing number of orientations add to more computation and it then takes more time to create the video. 
Not that 2nd (last channel - V) gives NaN output when getting the HOG output if the image is from 0-1. So, I had to scale the image to 0-255 to deal with this issue.

#### 3. Training classifier using the selected HOG features & Sliding Window Search

I trained a linear SVM. The features were normalized using standard scaler from sklearn. The data was randomly shuffled and splitted into training and test sets 1/5 ratio. The code is in the 16th cell of the ipython notebook

### Sliding Window Search

#### 1. Overlap, scale, start-stop

I decided to search with small windows size in around the center of the image and with larger windows size at the bottom of the image (As the cars near the camera appear to be bigger in the image). The scales of 1, 1,5, 2, 2.5 and 3 were seleted to make the pipeline robust for different scales of the car. The overlap of 0.75 was selected to make the computation faster. Then the images were converted to a heatmap, on which a threshold was applied to remove the noises. Sliding windows can be seen in the image below:

![alt text][image3]

### Heatmap Detection
Here is the heatmap for the test images

![alt text][image5]

And the bounding boxes after thresholding looks like this:

![alt text][image7]

---
## Methods

### Feature extraction and SVM training

Positive and negative sample data came from the [Udacity dataset](https://github.com/udacity/CarND-Vehicle-Detection), which contained 8799 images of vehicles viewed from different angles and 8971 images of non-vehicles (road, highway guardrails, etc.), all cropped and resized to 64x64 pixels. HOG, color histogram, and/or spatial features were extracted for each sample image. I experimented with a variety of color spaces (BGR, grayscale, HLS, HSV, Lab, Luv, YCrCb, YUV) and channels (any combination of 0, 1, 2) and HOG parameters (number of bins, cell size, block size, signed vs unsigned gradients). I tried both the scikit-image and OpenCV HOG implementations; the OpenCV function was roughly 5X faster than its scikit-image counterpart. I also experimented with varying numbers of color histogram bins and spatial feature sizes. Examples of a vehicle image and non-vehicle image are shown below alongside visualizations of their HOG gradient histograms.

![HOG_visualization](https://github.com/vaibhavhariaramani/hog--svm-vehicle-detector-and-Object-Detection/blob/master/images/hog_visualization.png)

After feature extraction, the feature vectors were scaled via `sklearn.preprocessing.StandardScaler`.

Finally, the sets of feature vectors were shuffled, then divided 75%/20%/5% into training, cross-validation, and test sets. I originally tried a 70%/20%/10% split but found that the classifier's performance on the test set (frequently >99%) was not necessarily indicative of its real-world effectiveness. Consequently, I chose to reduce the size of the test set.

The classifier itself was a `sklearn.svm.LinearSVC`. A number of parameters were tried for the SVC, including penalty, loss function, and regularization parameter. The training set was augmented with misclassifications from the cross-validation set and the classifier was retrained on this augmented training set before being evaluated on the test set, on which high accuracy (>99%) was frequently achieved.

### Object detection

A variable-size sliding window search was performed on each frame of the input video. Originally, I tried a fixed-size sliding window with an image pyramid to find objects at different scales, but this resulted in a relatively large number of false positives. Only the portion of the image below the horizon was searched to reduce computation and false positives (since there should be no vehicles above the horizon). An example of the variable-size sliding window can be seen below (as well as a link to a video of the technique).

![sliding_window](https://github.com/vaibhavhariaramani/hog--svm-vehicle-detector-and-Object-Detection/blob/master/images/sliding_window_example.png)
[Link to sliding window example video](https://youtu.be/9s7dUlmLVk4)

The rationale behind this sliding window approach is that vehicles in the vicinity of the horizon will appear smaller since they're farther away, while vehicles near the bottom of the image are closer and appear larger.

Due to window overlap, a single vehicle generally produced multiple overlapping detections:

![overlapping_detections](https://github.com/vaibhavhariaramani/hog--svm-vehicle-detector-and-Object-Detection/blob/master/images/sliding_window_detections.png)
[Link to video with raw detections drawn at each frame](https://youtu.be/RN2YW8y0qzY)

To fix this situation we’ll need to apply Non-Maximum Suppression (NMS), also called Non-Maxima Suppression.
[Link to implementation of Non-Maximum Supression (NMS) ](https://youtu.be/TfX6jtuPL0I)

## Results
### Video Implementation

#### 1. Video  Output
Here's a [link to my video result][https://youtu.be/rV89KM6izi8]


#### 2. Filters to remove false positive and combining rectangles

I recorded the positions of all detections in each frame. Then after every N (5) frames I did the following: 
1. Created heatmap of individual frames
2. Threshold heatmap of individual frames
3. Combinate heatmap of all N (5) frame
4. Threshold the accumulated heatmap. 
5. Used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and assumed each blob corresponded to a vehicle
6. Constructed bounding boxes to cover the area of each blob detected. 

All the frames are bboxes from the individual frames are inserted into queue and this way bboxes of last 5 frames are retained. This make the pipeline more robust. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
![alt text][image8]

---

[Link to video with raw detections drawn at each frame](https://youtu.be/RN2YW8y0qzY)

To fix this situation we’ll need to apply Non-Maximum Suppression (NMS), also called Non-Maxima Suppression.
[Link to implementation of Non-Maximum Supression (NMS) ](https://github.com/vaibhavhariaramani/Non-Maximum-Suppression-for-Object-Detection-in-Python)

---
# Resources 

To learn more about these Resources you can Refer to some of these articles written by Me:-

https://sites.google.com/view/geeky-traveller/computer-vision/histogram-of-oriented-gradients-and-object-detection

### Made with ❤️ by Vaibhav Hariramani
#### About me

I am an Actions on Google, Internet of things, Alexa Skills, and Image processing developer.
I have a keen interest in Image processing and Andriod development.
I am Currently studying at  Chandigarh University, Punjab.

You can find me at:-
[Linkedin](https://www.linkedin.com/in/vaibhav-hariramani-087488186/) or [Github](https://github.com/vaibhavhariaramani) .

Happy coding ❤️ .
