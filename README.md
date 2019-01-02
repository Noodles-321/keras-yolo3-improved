# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


CSDN博客地址：https://blog.csdn.net/sinat_26917383/article/details/85614247


# 1 数据准备
最简单是因为把数据整理成以下的样子就可以开始训练：

```
path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
path/to/img2.jpg 120,300,250,600,2
...
```

也就是：地址，xmin,ymin,xmax,ymax，类别ID然后空格下一个box，每张图一行。
例子：

```
images/images_all/86900fb6gy1fl4822o7qmj22ao328qv7.jpg 10,259,399,580,27
images/images_all/b95fe9cbgw1eyw88vlifjj20c70hsq46.jpg 10,353,439,640,29
images/images_all/005CsCZ0jw1f1n8kcj8m1j30ku0kumz6.jpg 75,141,343,321,27
```

----------


# 2 训练：
keras源码中有两段训练：

- 第一段冻结前面的249层进行迁移学习（原有的yolov3）
- 第二段解冻全部层进行训练

笔者自己的训练数据集是专业领域的图像，所以基本第一阶段的迁移学习阶段没啥用，因为与原有的yolov3训练集差异太大，如果你也是，请直接开始第二段或者重新根据[darknet53](https://pjreddie.com/media/files/darknet53.conv.74)训练。
那么这边就有三样可能需要预下载的模型：

- yolo_weights.h5 预训练模型（用作迁移）
```python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5```
- darknet53.weights （用作重新训练）
```wget https://pjreddie.com/media/files/darknet53.conv.74```
- yolo.h5 （yolov3-VOC训练模型，可以直接用来做预测 ）
```python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5```

来看看训练时候需要的参数：

```
    class yolo_args:
        annotation_path = 'train.txt'
        log_dir = 'logs/003/'
        classes_path = 'model_data/class_file_en.txt'
        anchors_path = 'model_data/yolo_anchors.txt'
        input_shape = (416,416) # multiple of 32, hw
        # 608*608  416*416  320*320
        val_split = 0.1
        batch_size = 16
        epochs_stage_1 = 10
        stage_1_train = False
        epochs_finally = 100
        finally_train = True
        weights_path =   'logs/003/ep009-loss33.297-val_loss32.851.h5'# 可以使用'model_data/tiny_yolo_weights.h5' 也可以使用tiny_yolo的：'model_data/yolo_weights.h5'

        
        
    # train
    _main(yolo_args)
```
 - annotation_path就是数据集准备的txt
 - log_dir ，Model存放地址，譬如：`events.out.tfevents.1545966202`、`ep077-loss19.318-val_loss19.682.h5`
 - classes_path ，分类内容
 - anchors_path ，yolo anchors，可自行调整，也可以使用默认的
 - input_shape ，一般是416

 - `epochs_stage_1 = 10 `和 `stage_1_train = False`，是同一个，也就是是否进行迁移学习（`stage_1_train` ），要学习的话，学习几个epoch（`epochs_stage_1` ）
 - `epochs_finally = 100 `和 `finally_train = True` ，是，是否进行后面开放所有层的学习（`finally_train` ），学习几个epoch（`epochs_finally `）
 - weights_path ，调用model的路径


**这里需要注意：**
如果要在之前训练基础上，追加训练，一般要把batch_size设置小一些，然后加载之前的权重。

----------

# 3 预测：
来看一个简单的预测
```
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

yolo_test_args = {
    "model_path": 'model_data/yolo.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "score" : 0.3,
    "iou" : 0.45,
    "model_image_size" : (416, 416),
    "gpu_num" : 1,
}


yolo_test = YOLO(**yolo_test_args)
image = Image.open('images/part1/path1.jpg')
r_image = yolo_test.detect_image(image)
r_image.show()
```
直接返回的是带框的图片，如果你要输出boxes，可以自己改一下`detect_image`函数。

----------
##########################################################
##############          原项目内容           ##############
##########################################################

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
