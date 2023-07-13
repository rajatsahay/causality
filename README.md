# causality

~~ From my notes, this does not run on a Mac. Something to do with the way OpenCV is built for Macs. <br>
I haven't tried this on a Windows machine, but to use it with linux, you would have to build OpenCV from scratch rather than using ```pip``` or ```conda-forge```. <br>
<br>
The instructions can be found [here](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html). ~~

<br> <br>

<h3> Instructions for running: </h3>

Please install [PyTorch](https://pytorch.org/get-started/locally/) as per your system requirements. 

``` 
pip install -r requirements.txt 
```

Change the paths at the following lines: <br>
    Line 21: Add API Key <br>
    Line 65: Add path to the annotation directory <br>
    Line 86: Add path to the proposal directory <br>
    Line 161: Add path to directory with all video files <br>
    Line 168: Add path to the ```validation.json``` file <br>

All directories can be downloaded from the [CLEVRER dataset page](http://clevrer.csail.mit.edu/)

<br><br>

Coordinates (for future use) will be stored in the gpt_coords directory, while question-answer pairs (for evaluation) will be stored in gpt_logs.

