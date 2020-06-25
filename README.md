# Welcome to my AI challenge.

I tried to solve task 1.1 of the [abid challenge](https://github.com/silverbottlep/abid_challenge
). While trying to solve this challenge I came across a couple of small issues.

## The issues I have faced.
- I don't have an Amazon account.
	- I solved this by making a [script](https://github.com/Vepnar/AI-challenge/blob/master/bucket-downloader.sh) that would download every file from the AWS Bucket with HTTP requests. This was rather slow but it worked.
- The images are not all the same size and are in RGB.
	- I solved it by using this [script](https://github.com/Vepnar/AI-challenge/blob/master/image_processor.py). This would resize all images and make them grey with OpenCV.
- Overfitting.
	- I first downloaded around 1200 images of 5 types. I quickly noticed that my model has a low training loss and a high validation loss. This means it is overfitting and I need more data. <br>
Training accuracy:
![Training accuracy one](https://raw.githubusercontent.com/Vepnar/AI-challenge/master/pictures/train1.png)Validation accuracy:
![Validation accuracy one](https://raw.githubusercontent.com/Vepnar/AI-challenge/master/pictures/validation1.png)

- Unbalanced data set.
	- Now I've downloaded a bigger dataset but it is very unbalanced like shown below. For analysing the data I've used the following [script](https://github.com/Vepnar/AI-challenge/blob/master/image_analyzer.py).
![unbalanced data](https://raw.githubusercontent.com/silverbottlep/abid_challenge/master/figs/stats.png)      
-  Training on the CPU is slow and I can't downgrade my CUDA drivers.
	- That is why started [training](https://github.com/Vepnar/AI-challenge/blob/master/training.py) on my computer with [this](https://github.com/Vepnar/AI-challenge/blob/master/Dockerfile) and [this](https://github.com/Vepnar/AI-challenge/blob/master/docker-compose.yml) docker file.
- Optimising the model itself is rather slow.
	- That is why I decided to make it automated with the following [script](https://github.com/Vepnar/AI-challenge/blob/master/automated_trainer.py).
- Tensorflow takes too much VRAM.
	- That is why I created my own  batch generator. This made it possible to train with somewhat bigger neuralnetworks but it hugely reduced the training speed.

# Limitations
After trying so many different models I got problems with my system. Some network like [AlexNet](https://en.wikipedia.org/wiki/AlexNet) are just too heavy for my system to try. I sadly have to abandon this project because of the limitations on my system.

## Things I've tried
- Different amounts  / sizes of 2D convolutional layers.
- Different amounts  / sizes of deep layers.
- Different activation functions for example: (relu) Rectifier, softmax, (tanh) Hyperbolic functions, sigmoid.
- Different kernel sizes.
- Dropouts.
- Batch normalization.
- Alexnet

![Alexnet](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimage.slidesharecdn.com%2Fpydatatalk-150729202131-lva1-app6892%2F95%2Fdeep-learning-with-python-pydata-seattle-2015-35-638.jpg%3Fcb%3D1438315555&f=1&nofb=1)
