# Domain Adaptation for anime face detection
This is an implementation of domain adaptation for anime face detection on Pytorch.

We referred [hdjsjyl/face-faster-rcnn.pytorch](https://github.com/hdjsjyl/face-faster-rcnn.pytorch).

## Preparation
First of all, clone the code
```
git clone https://github.com/kanosawa/anime-face-faster-rcnn-da.pytorch.git
```

Then, create a folder:
```
cd anime-face-faster-rcnn-da.pytorch && mkdir data
```

### Data Preparation
1. [WIDER Face dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
2. [CelebA dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) : img_align_celeba.zip
3. [animeface-character-dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip)

Download above data and extract to **data** folder as below structure.
   * data
     * WIDER2015
       * eval_tools
       * wider_face_split
       * WIDER_test
       * WIDER_train
       * WIDER_val
     * img_align_celeba
     * animeface-character-dataset

### Pretationed Model
Download [VGG16](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0) and put them into the data/pretrained_model/

### Compilation
```
cd lib
python setup.py build develop
```

## Train
```
python trainval_net.py 
```

## Demo
Put images to **images** folder and execute below command.
```
python demo.py --checksession $SESSION --checkepoch $EPOCH --checkpoint $POINT
```
if you want to use output/vgg16/wider_face/faster_rcnn_1_3_6439.pth,
substitute $SESSION=1, $EPOCH=3, $POINT=6439