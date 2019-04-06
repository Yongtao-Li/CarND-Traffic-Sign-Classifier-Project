# **Traffic Sign Recognition** 

## Yongtao Li

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_histograms.png "histograms"
[image2]: ./examples/example_image.png "example image"
[image3]: ./examples/example_gray_image.png "example gray image"
[image4]: ./examples/model_train.png "model train"
[image5]: ./test_images/11_Rightofway.jpg "Traffic Sign 1"
[image6]: ./test_images/17_Noentry.jpg "Traffic Sign 2"
[image7]: ./test_images/25_RoadWork.jpg "Traffic Sign 3"
[image8]: ./test_images/33_RightOnly.jpg "Traffic Sign 4"
[image9]: ./test_images/12_PriorityRoad.jpg "Traffic Sign 5"
[image10]: ./test_images/14_Stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Yongtao-Li/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_YL.ipynb). The HTML version of the notebook could be found [here](https://github.com/Yongtao-Li/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_YL.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

first of all, I loaded the data sets using pickle files

```python
training_file = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
```

then I used numpy to calculate the statistics of the traffic signs data set

```python
n_train = y_train.shape[0]
n_validation = y_valid.shape[0]
n_test = y_test.shape[0]
image_shape = X_train.shape[1:3]
n_classes = len(np.unique(y_train))
```

The results look like the following:

```python
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot of histograms for train, validation and test data. It shows that we have comparable amount of pictures for various classes among these three data sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used opencv function to convert trafic signs data to gray scale. The example code is as following.

```python
def preprocess(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray[:, :, newaxis]
    return img_gray
```

As you could see the example below before and after the grayscale convention. However I didn't end up using it since my final model is already doing a decent job on validation accuracy using RGB images.

| example image      |example gray image | 
|:------------------:|:-----------------:|
| ![alt text][image2]| ![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started my model like LeNet architecture, but not getting enough validation accuracy. So I added more convolutional layers into the final model as following.

| Layer         		|     Description	        					| 
|:--------------------- |:--------------------------------------------- | 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3x8     | 1x1 stride, valid padding, outputs 30x30x8 	|
| RELU					| activation layer      						|
| Convolution 3x3x16    | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					| activation layer      						|
| Max pooling	      	| 2x2 stride, outputs 14x14x16  				|
| Convolution 3x3x32    | 1x1 stride, valid padding, outputs 12x12x32   |
| RELU					| activation layer      						|
| Convolution 3x3x32    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					| activation layer      						|
| Max pooling	      	| 2x2 stride, outputs 5x5x32    				|
| Convolution 3x3x32    | 1x1 stride, valid padding, outputs 3x3x32     |
| RELU					| activation layer      						|
| Flatten               | outputs vector 288                            |
| Fully connected		| outputs 120									|
| Fully connected		| outputs 84									|
| Fully connected		| outputs 43									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train this model, I used the same Adam optimizer as in the LeNet lab. I kept learning rate as 0.001, batch size as 256 and number of epochs as 50.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95% 
* test set accuracy of 94%

Initially I was using LeNet 3 layers and default hyperparameters from previous LeNet lab. As in the following snapshot, it didn't reach desired validation accuracy. I thought I need more layers because I have 43 classes to classify, 4 times more than the previous LeNet lab. So I added more layers to the final model which results in better validation accuracy. I also tried to increase the batch size from 128 to 256 and epoch from 10 to 50. The final model trained shows 95% validation accuracy and 94% accuracy on test data set.

![alt text][image4] 

I also attached the spreadsheet that generates the plots above [here](https://github.com/Yongtao-Li/CarND-Traffic-Sign-Classifier-Project/blob/master/model_training_notes.xlsx).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

You could tell that all these images are difference sizes and our model needs input for a specific size for inputs.

```python
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
```

So I used opencv function to resize them to be exact 32x32, therefore I could leverage my final model directly to get predictions.

```python
res = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
```

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

By restoring previously trained model, I calculated the predictions and probabities for test images. By picking the biggest probability, the predictions did an amazing job of guessing all test images right! This is a little bit higher than previous 94% test accuracy, but we only have a few new test images tested here.

```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    predict = sess.run(logits, feed_dict = {x: X_test_img})
    probs=sess.run(tf.nn.softmax(predict))

print ('class ID predictions are: ', np.argmax(probs, 1))
```

Here are the results of the prediction:

| Image			        |     Prediction  | 
|:----------------------|:---------------:| 
| 11 Right of Way  		| 11   			  | 
| 17 No Entry 			| 17 			  |
| 25 Road Work			| 25			  |
| 33 Right Only    		| 33			  |
| 12 Priority Road  	| 12      		  |
| 14 Stop               | 14              |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following code shows how to print out the top 5 softmax probabilities for each image once predictions are made.

```python
with tf.Session() as sess:
    top_5 = sess.run(tf.nn.top_k(tf.constant(probs), k=5))
    print (top_5)
```
The top five soft max probabilities for each prediction are as following. It seems the model is very confident for the predictions since the highest probability is very close to 1.

| Top 5 Probability         	                                          |     Prediction  |  
|:-----------------------------------------------------------------------:|:---------------:|
| 9.9999940e-01, 3.2776660e-07, 2.9114602e-07, 1.8454999e-08,4.2180286e-09| 11   			|  
| 1.0000000e+00, 1.7869048e-14, 9.4393788e-19, 4.0660042e-22,3.7322572e-22| 17 			    |
| 9.9862385e-01, 7.8410358e-04, 5.2689260e-04, 2.2095543e-05,1.4150333e-05| 25			    |
| 1.0000000e+00, 8.9690539e-09, 7.5393569e-17, 2.5752147e-18,1.2876239e-18| 33			    |
| 1.0000000e+00, 3.0966429e-21, 2.8721633e-22, 4.7103988e-26,9.1462706e-27| 12      		|
| 9.9999905e-01, 9.4512990e-07, 3.9446906e-08, 1.5406231e-08,8.9201119e-10| 14              |


