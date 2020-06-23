# AI-Challenge

I'm trying to solve this challenge [this]("https://github.com/silverbottlep/abid_challenge") without an amazon AWS account.

## First problem
I do not have an Amazon AWS account and I can not create one because it requires credit card information which I dont have. So I came up with an alternative.
[this script]("https://github.com/Vepnar/AI-challenge/blob/master/bucket-downloader.sh") Can download all files and metadata from the amazon bucket to my computer without the need of an amazon bucket

## Second thing to do.
We need to preprocess our images to be the same size with OpenCV2.

I used `image_processor.py` to make all images 50x50 pixels and made them grayscale.

This will reduce the amount of computing power I need to train the model.

# How to run this on your system
1. Clone my github repository on your system.
`git clone https://github.com/Vepnar/AI-challenge.git`
2. Download all requirements
`sudo apt-get install wget python3 python3-opencv2`
3. CD into into my github repository
`cd AI-challenge`
3. Make the bash file executable
`chmod +x bucket-downloader.sh`
4. Download all images (This might take a while)
`./bucket-downloader.sh`
5. Process all downloaded images
`python3 image_processor.py`

