# FasterRCNN_simple
First step, remember to use git pull <br>
Second step, download data: [Download Link](https://drive.google.com/drive/folders/1eP7FtPaWfJ5zLdcsZYl6eyn5EYixkFn8?usp=sharing) <br>
Third step, get the resnet50 cehckpoint <br>
Forth step, To get the proposals with no-background class, run proposal_printer.py. This code will save 6 images on you machine that show the proposals with no-background class in blue boxes and shows the associated bounding box/boxes in red color. These images are stored with names starting with: no_background_200_props.<br>
Fifth step, To train the model, run training_model.py. This will save the model with the name model.pth <br>
The loss curves would be stored in files: (Training) tlc.png, tlr.png, tlt.png, (Validation) vlc.png, vlr.png and vlt.png. <br>
Our top 20 proposals would be stored in png files with names starting with: top20props. <br>
The NMS images would be stored under png files with names: beforeNMS and afterNMS. <br>
The visualization of original images would be stored in png files with names starting with: vis. <br>
The ground truth images would be stored in png files with names starting with: gt. <br>
If there is no model.pth present, then dataset.py will train a model and store the new model as model.pth.<br>
