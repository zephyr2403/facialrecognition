# About
Facial Recognition Utilizing Local Binary Pattern Histogram

Local binary patterns (LBP) is a type of visual descriptor used for classification.
The LBP feature vector, in its simplest form, is created in the following manner:

* Divide the examined window into cells (e.g. 16x16 pixels for each cell).
* For each pixel in a cell, compare the pixel to each of its 8 neighbors (on its left-top, left-middle, left-bottom, right-top, etc.). Follow the pixels along a circle, i.e. clockwise or counter-clockwise.
* Where the center pixel's value is greater than the neighbor's value, write "0". Otherwise, write "1". This gives an 8-digit binary number (which is usually converted to decimal for convenience).
* Compute the histogram, over the cell, of the frequency of each "number" occurring (i.e., each combination of which pixels are smaller and which are greater than the center). This histogram can be seen as a 256-dimensional feature vector.
* Optionally normalize the histogram.
* Concatenate (normalized) histograms of all cells. This gives a feature vector for the entire window.

### Requirements:
* Python
* opencv-contrib

### Usage:

* Select whether you want to train the model or not.
* From train it will capture 300 images(less than 1 min) of your face and save it into `./faces`
* These images will be used for training the model.  

**Note:** If you have no training data (i.e no images in `./faces`) and you choose not to train model then program will throw an error.
