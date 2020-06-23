# AI-Challenge.

I'm trying to solve this challenge [this]("https://github.com/silverbottlep/abid_challenge") without an amazon AWS account. On a rather old system.

## First problem.
I don't have an Amazon AWS account and I can not create one. That's why I created this script as an alternative.
[this script]("https://github.com/Vepnar/AI-challenge/blob/master/bucket-downloader.sh") will download all files and metadata from the amazon bucket to my computer without the need of an amazon bucket

## Second thing I did.
We need to preprocess our images to be the same size as OpenCV2.

I used `image_processor.py` to make all images 50x50 pixels and made them grayscale.

This will reduce the amount of computing power I need to train the model.

### Third problem.
I didn't download enough images to train my machine-learning model. 

### Fourth problem.
The data needs to be clean and is not balanced.

### Fifth problem.
Tweaking the dataset takes a lot of time & tensorflow doesn't release the used RAM/VRAM.

# How to run this on your system.
1. Clone my GitHub repository on your system.
`git clone https://github.com/Vepnar/AI-challenge.git`
2. Download requirements for this application.
`sudo apt-get install wget python3 python3-opencv2 tensorflow`
3. Go into the cloned GitHub repository.
`cd AI-challenge`
4. Update the permissions bash script to make it executable.
`chmod +x bucket-downloader.sh`
5. Download all images from the AWS bucket.
`./bucket-downloader.sh`
6. Now you need to process all the downloaded images.
`python3 image_processor.py`
7. Now you're ready to start the training.
`python3 training.py`   

