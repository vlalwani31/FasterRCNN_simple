# FasterRCNN_simple
First step, remember to use git pull <br>
Second step, download data: [Download Link](https://drive.google.com/drive/folders/1eP7FtPaWfJ5zLdcsZYl6eyn5EYixkFn8?usp=sharing) <br>
Third step, Run the dataset.py file. This will train the model if model.pth is not present in the same folder as dataset.py. It will also perform pointwise Accuracy and before and after NMS inference. It also calculates the histogram for the assignment. <br>
All the outputs would be saved in image format. <br>
The loss curves would be stored in files: (Training) tlc.png, tlr.png, tlt.png, (Validation) vlc.png, vlr.png and vlt.png. <br>
Our top 20 proposals would be stored in png files with names starting with: top20props. <br>
The NMS images would be stored under png files with names: beforeNMS and afterNMS. <br>
The visualization of original images would be stored in png files with names starting with: vis. <br>
The ground truth images would be stored in png files with names starting with: gt. <br>
If there is no model.pth present, then dataset.py will train a model and store the new model as model.pth.<br>
