![Demo](images/outputvideo_10.gif)
<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml">
  <body class="c14">
    (program video output)
    <h1 class="c26" id="h.qze4dlw21i4d">
      <span class="c16">Vehicle Detection and track</span>
    </h1>
    <h2 class="c24" id="h.52013ybplk45">
      <span class="c6">Overview:</span>
    </h2>
    <p class="c5">
      <span class="c0">In this project I will detect and track vehicles on the road by using Histogram of Oriented Gradients(HOG) for features extraction and supervised machine learning algorithm for detection. </span>
    </p>
    <p class="c5">
      <span class="c0"> SVM is a supervised machine learning algorithm which can be used for classification. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs. Different from matching templates, SVM can generaize the detection of the car. 
      </span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <hr/>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <h2 class="c24" id="h.yd5ehac7qkvb">
      <span class="c6">Project pipeline </span>
    </h2>
    <ul class="c19 lst-kix_nz5mknjr6itm-0 start">
      <li class="c5 c13">
        <span class="c0">Histogram of Oriented Gradients (HOG)</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-1 start">
      <li class="c5 c7">
        <span class="c0">Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-0">
      <li class="c5 c13">
        <span class="c0">Image augmentation</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-1 start">
      <li class="c5 c7">
        <span class="c0">Normalize features and randomize a selection for training and testing.</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-0">
      <li class="c5 c13">
        <span class="c0">Train a linear svm classifier to search for vehicles in images</span>
      </li>
      <li class="c5 c13">
        <span class="c0">Implement a sliding-window technique</span>
      </li>
      <li class="c5 c13">
        <span class="c0">Heat map</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-1 start">
      <li class="c5 c7">
        <span class="c0">Used heat map to reject outliers as it collects recurring detections frames and followed by detected vehicles.</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-0">
      <li class="c5 c13">
        <span class="c0">Bounding box</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-1 start">
      <li class="c5 c7">
        <span class="c0">Estimates a bounding box for vehicles detected.</span>
      </li>
    </ul>
    <ul class="c19 lst-kix_nz5mknjr6itm-0">
      <li class="c5 c13">
        <span class="c0">Run your pipeline on a video stream</span>
      </li>
    </ul>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <hr/>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <h1 class="c27" id="h.n0rwgthx2661">
      <span class="c16">Data set</span>
    </h1>
    <p class="c5">
      <span>Data set is provided by Udacity. Here are links to the labeled data for</span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip&amp;sa=D&amp;ust=1502661329599000&amp;usg=AFQjCNFEs4aO-461tf8S-w-M8DY1V8iiLA"> vehicle</a>
      </span>
      <span> and</span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip&amp;sa=D&amp;ust=1502661329599000&amp;usg=AFQjCNHO_a74Tutee1XeMg1ZszcbNrBZxg"> non-vehicle</a>
      </span>
      <span class="c0">. Each folder contains images and a csv file containing all the labels and bounding boxes. To add vehicle images to your training data, you'll need to use the csv files to extract the bounding box regions and scale them to the same size as the rest of the training images.</span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 274.50px;">
        <img alt="" src="images/image2.png" style="width: 624.00px; height: 274.50px; margin-left: 0.00px; margin-top: 4.99px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""/>
      </span>
    </p>
    <h2 class="c15" id="h.keitt150eipm">
      <span class="c6">Histogram of Oriented Gradients (HOG)</span>
    </h2>
    <p class="c5">
      <span class="c0">Gradient Features - It allows you to get the shape of the object, so later you can create an independent signature from color and size. In general terms the gradient allow you recognize the direction (gradient direction) in which the color pixels are changing. This technique is call HOG Features. (histogram of oriented gradients) </span>
    </p>
    <p class="c5 c25">
      <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 160.00px;">
        <img alt="" src="images/image3.png" style="width: 624.00px; height: 160.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""/>
      </span>
    </p>
    <h4 class="c23" id="h.b0zfj726mzs6">
      <span class="c11">#HOG call from the search car function</span>
    </h4>
    <p class="c5">
      <span class="c0">hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)</span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span class="c0">The code for this step is contained in the first code cell of the IPython notebook (under Search Car function,  get_hog_features can be found in lines 107 to 120 lesson_Functions.py)</span>
    </p>
    <p class="c5">
      <span>The bottom image is an example of the data set used for classification. Example of</span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/&amp;sa=D&amp;ust=1502661329602000&amp;usg=AFQjCNFVj2fw4qb-GTJYnI1isBrrOn0rJg"> HOG</a>
      </span>
    </p>
    <h3 class="c10" id="h.r6t83uk6himv">
      <span class="c1">Tuning HOG parameters</span>
    </h3>
    <p class="c5 c8">
      <span class="c3"/>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span class="c0">I started with few samples of vehicles and non-vehicles images, to explored parameter tuning. After tuning paraments I ran a classifier to identify how the HOG features changed the predictions of the system.  </span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span class="c0">The paraments that were giving me positive results are YCrCb color space and HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):</span>
    </p>
    <p class="c5">
      <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 448.00px; height: 236.00px;">
        <img alt="" src="images/image4.png" style="width: 448.00px; height: 236.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""/>
      </span>
    </p>
    <h2 class="c24" id="h.9hexc493s1yq">
      <span class="c6">Sliding Window Search</span>
    </h2>
    <h3 class="c10" id="h.clwjk33yg8a8">
      <span class="c1">Region of search:</span>
    </h3>
    <p class="c5">
      <span class="c0">The region of the image to search for a car, starts a little over halfway down the image (y_start_stop=[int(image.shape[0]/2), image.shape[0]-50]). The image cut allow us to search cars on the road, and not on the sky or trees. </span>
    </p>
    <h3 class="c10" id="h.c7odt6umacds">
      <span class="c1">Tiles Size:</span>
    </h3>
    <p class="c5">
      <span class="c0">The size of the tiles was determined to be 64 so it can match the sampling rate 8 cells and 8 pix per cell.</span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <h2 class="c24" id="h.vgcxhj2qyv3q">
      <span class="c6">Sliding window implementation search</span>
    </h2>
    <p class="c5">
      <span>To addess the variance of car size across the video frame, instead of changing the window size, I resize the origial image to multiple size. This image scale method resizes the original image to fit a window. This technique was borrowed from Udacity's Self-driving class. </span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span>(implementation can be found under find_cars function in </span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=https://ec2-52-53-212-62.us-west-1.compute.amazonaws.com:8888/notebooks/notebook/CarND-Vehicle-Detection/source_code/Vehicle_Detection.ipynb&amp;sa=D&amp;ust=1502661329608000&amp;usg=AFQjCNF7yUn9mq3AFwFGyQJUJyyB2NadMw">Vehicle_Detection.ipynb</a>
      </span>
      <span class="c0">)          </span>
    </p>
    <h2 class="c24" id="h.6py0tggq0q0w">
      <span class="c6">Training a classifier</span>
    </h2>
    <p class="c5">
      <span>The classifier is trained using the HOG features. The </span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html%23sklearn.svm.SVC&amp;sa=D&amp;ust=1502661329610000&amp;usg=AFQjCNERRouFwxgJ0LBwkGpQcqcnJO_FDQ">user guide</a>
      </span>
      <span class="c0"> from sckit-learn to identify the appropriate classifier and settings. My setting for the classifier is:</span>
    </p>
    <p class="c5">
      <span class="c0">  svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, decision_function_shape='ovr', random_state=None)</span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c5">
      <span>The input data with best results came from a concatenation of HOG features from 'YCrCb' color space (8 cells per block). My classifier has a kernel ‘rbf’ and gramma ’auto’. Implementation of the pipeline can be found under function pipeline in </span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=https://ec2-52-53-212-62.us-west-1.compute.amazonaws.com:8888/notebooks/notebook/CarND-Vehicle-Detection/source_code/Vehicle_Detection.ipynb&amp;sa=D&amp;ust=1502661329611000&amp;usg=AFQjCNEq0BknDzwmo7jKG4fF25zYbS9BDA">Vehicle_Detection.ipynb</a>
      </span>
      <span class="c0">. </span>
    </p>
    <h3 class="c28" id="h.spsnkny3w45d">
      <span class="c1">Overlapping Boxes </span>
    </h3>
    <p class="c5">
      <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 194.67px;">
        <img alt="" src="images/image1.png" style="width: 624.00px; height: 194.67px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""/>
      </span>
    </p>
    <p class="c12">
      <span class="c0">To smooth overlapping and filter false positives of the image I use a detection overlap threshold. Overlap treshold was manualy selected to two. </span>
    </p>
    <p class="c5 c18">
      <span class="c22">heatmap = apply_threshold(heatmap, 2)</span>
    </p>
      <span>Implementation of the pipeline can be found under function pipeline and find_car functions in </span>
      <span class="c2">
        <a class="c4" href="https://www.google.com/url?q=https://ec2-52-53-212-62.us-west-1.compute.amazonaws.com:8888/notebooks/notebook/CarND-Vehicle-Detection/source_code/Vehicle_Detection.ipynb&amp;sa=D&amp;ust=1502661329614000&amp;usg=AFQjCNGPAPvp3AcBGZtwD182vEz111V4fQ">Vehicle_Detection.ipynb</a>
      </span>
      <span class="c0">. </span>
    </p>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <p class="c20">
      <span class="c0"> Below is a visualization of detection heat-map </span>
    </p>
    <h3 class="c29" id="h.jq81i2g4wqwy">
      <span class="c30">Here the resulting bounding boxes are drawn onto the last frame in the </span>
      <span class="c3">1. </span>
      <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 603.14px; height: 939.50px;">
        <img alt="" src="images/image5.png" style="width: 603.14px; height: 939.50px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""/>
      </span>
    </h3>
    <p class="c5 c8">
      <span class="c0"/>
    </p>
    <h2 class="c24" id="h.9z9142o9x1l1">
      <span>Future Implementations</span>
    </h2>
    <ul class="c19 lst-kix_5r8avx40xan8-0 start">
      <li class="c5 c13">
        <span class="c0"> Non-max suppression allows us to remove weaker matches to produce uniquedetections. </span>
      </li>
      <li class="c5 c13">
        <span class="c0"> CNN techniques for detection </span>
      </li>
      <li class="c5 c13">
        <span class="c0">Temporal sequence neural net to keep track of previews results like RNN </span>
      </li>
    </ul>
  </body>
</html>
