**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.jpg
[image2]: ./output_images/HLS_HOG_car.jpg
[image3]: ./output_images/YCrCb_HOG_car.jpg
[image4]: ./output_images/detected_vehicles_of_boxes.jpg
[image5]: ./output_images/heatmap_detected_vehicles.jpg
[image6]: ./output_images/label_detected_vehicles.jpg
[image7]: ./output_images/Bounding_box_detected_vehicles.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

We can see that the hog images of YCrCb transform image contain more generated vectors for all three channels than HLS transform image and the former show slightly higher prediction accuracy than the latter. So I will use YCrCb transform to generate hog vector images in my following code.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and find orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) will give best prediction accuracy when fitting the generated features to classifier. I also tried pixels_per_cell=(4, 4) but that caused my computer crashed as this exhausting all resources of my computer. So I can only tweak the parameters within generated features of size around 8000.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the fifth and sixth code cells of the IPython notebook.  

I trained a linear SVM using `svc = LinearSVC()` and before fitting into classifier, I normalized input features with `StandardScaler`. After training, I saved the model into file 'trained_svc.pkl' using sklearn.externals.joblib(in seventh code cell of IPython notebook).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to apply the method in the course which define a single function that can extract features using hog sub-sampling and make predictions. This will be a easy way to extract valid boxes rather than machinery search method introduced in course. This part is contained in function `get_box_list` of eighth code cell. The output image will be display together in the next part. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There are two potential points where failure could occur.

One is the classifier. Although I got more than 0.98 for test accuracy of my classifier, it still could give false positive predictions on project videos. One main reason is the classifier was not fed with enough images especially with images similar to project video so the classifier cannot get sense for car or non-car objects in project videos. Of course I can add some preprocessing to avoid some difficult light conditions. Thus the possible way to inprove this will be capture more images similar to project video or use udacity database.

Another is the pipeline. As it occurs often where the classifier fails to give 'right' bounding boxes or even cannot detect vehicles in some difficult light conditions, I add a tracking list and a snippet of code(in function `draw_labeled_bboxes` of 8th code cell) to address the former one but I didn't handle the latter. One possible way could be add tracing queue of detected boxes of past consecutive frames like in previous project Advanced Lane Line. But I didn't have enough time to implement all this as I was off the course during the new year holiday backing to China to see my parents for 15 days.
