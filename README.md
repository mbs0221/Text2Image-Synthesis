# Text2Image-Synthesis
C-GAN, TextEncoder, COCO-2014

Install axel download accelerator
~~~
$ sudo apt-get install axel
$ axel -n 10 http://images.cocodataset.org/zips/train2014.zip
$ axel -n 10 http://images.cocodataset.org/zips/val2014.zip
$ axel -n 10 http://images.cocodataset.org/zips/test2014.zip
$ axel -n 10 http://images.cocodataset.org/annotations/annotations_trainval2014.zip
$ axel -n 10 http://images.cocodataset.org/annotations/image_info_test2014.zip
~~~
Create folders
~~~
$ mkdir coco-2014
$ mkdir coco-2014/images/
$ mkdir coco-2014/resized/
$ mkdir coco-2014/annotations/
~~~
Run python script
~~~
$ pip3 install -r requirements.txt
$ python3 utils/resize_image.py --source=coco-2014/images --target=coco-2014/resized/
$ run_with_gpu.py --root=../coco-2014 --sample_interval=10000 --cuda_id=1 --batch_size=48
~~~
