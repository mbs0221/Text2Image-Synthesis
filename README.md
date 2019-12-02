# Text2Image-Synthesis
C-GAN, TextEncoder, COCO-2014

Install Axel Download Accelerator
~~~
$ sudo apt-get install axel
~~~
Uncompress files
~~~
$ mkdir coco-2014
$ mkdir coco-2014/images/
$ unzip train2014.zip coco-2014/images/
$ unzip val2014.zip coco-2014/images/
$ unzip test2014.zip coco-2014/images/
$ mkdir coco-2014/annotations/
$ unzip annotations_trainval2014.zip coco-2014/annotations/
$ unzip image_info_test2014.zip coco-2014/annotations/
~~~
Run python script
~~~
$ pip3 install -r requirements.txt
$ python3 text2image.py
~~~