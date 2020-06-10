# Face Detection

## Introduction

This mini-project was done using the incredible package, that is OpenCV packages. I've followed the instruction on [this](https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348) website, he explain the workflow and "how to do" in each work-phase in a comprehensive way, I recommend that you take the time to look at the primary source. Only in this case, I was used my face to be detected. The workflow is represented by the following figure

![workflow](FaceRecogBlock.png)

As a part of the introduction, I will explain briefly what is it in each phase

1. Phase 1 : Data Gathering

This phase is intended to take the face image which want to be detected. HaarCascade detector file is used to crop and obtain the right face image. This phase is coded in the file `facial.py`. You can directly execute the file code `facial.py` by `python facial.py` in the terminal, this will record your face as many as 30 face images saved in `dataset/` folder. You can change the number of face images by changing the `count >= 30` in the code by any number you want.

2. Phase 2 : Train the Recognizer

This phase will feed the face image into the recognizer. The recognizer plays the same role as in the training phase in feed forward network where the network will take a unique features in the input to furhter used as prediction. This phase is coded in the file `recognizer.py` and the trained file is saved as `.yml` in `trainer/` folder.

3. Phase 3 : Recognizer

Finally, this phase is used to predict each frame of image.

## Result
The result is displayed in the following figure

![output](output.gif)
