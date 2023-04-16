# [[Raw域去噪]ISP的流程及raw图片处理](https://github.com/AlexiFeng/gitblog/issues/11)

去年年底的时候做了一个AI-ISP的项目，我负责的是raw域的去噪。与我们常见的ai算法不同，它是直接作用于相机原始数据的，更偏向底层数据处理，这也让我在做的过程中也对ISP部分有了一点了解。在这里写下一个入门笔记，仅供参考。

---

从宏观角度来讲，我们看到的照片通常经历了如下一个流程：光通过镜头打到CMOS传感器上-》得到原始数据（Bayer Mosaic）-》Demosaic-》白平衡，Gamma校正...-》出图

## 光打到CMOS传感器上
这个地方我特意提到了光打到CMOS传感器上，因为这是radiation noise形成的主要原因，而这与我后面要做的项目密切相关。
## Bayer Raw
raw格式文件通常是照片的原始文件，它是Bayer Mosaic的，可以简单理解为它并不是以RGB格式存储的，而是以RGGB/BGGR/RGBG...等方式进行存储的。当拍摄彩色图像的时候，最朴素的采样逻辑是用多块单色滤镜（红绿蓝三原色）拍摄单色图像，然后组合成一副彩色图像。但是这样造假高，而且必须要保证滤镜的位置完全对齐，不然一旦有像素偏差就会出现重影。Bayer则只用了一块滤镜，在不同位置设置不同的颜色，**由于人眼对绿色比较敏感，所以绿色用的更多**（这也解释了为什么Bayer Raw的格式里面都有两个Green,同时解释了demosaic之后的图片是绿色的）。
RGGB这只是滤色矩阵的编码方式，指的是一个2*2的像素方阵里各个颜色的顺序。直接附上一张图比较清晰。
![image](https://user-images.githubusercontent.com/16517113/232278729-dc7dbfb2-b489-4c11-b538-50afc541ff0a.png)

## Demosaic
为了将Bayer Raw重建为我们熟悉的RGB图像，需要对图像中的每一个像素点进行插值, 利用其周围像素点的色彩值来估计出缺失的另外两个色彩值, 最终得到一个每个像素点包含红、绿、蓝三个像素值的全彩色图, 这个过程就叫做Demosaic。如果使用了与滤色矩阵不适配的demosaic方法将会转出很奇怪的结果，轻则颜色错误，重则全是马赛克。opencv里有demosaic算法可以直接调用。这个阶段之后，图片就成为我们平时场景的格式了。在这个基础上再进行awb等操作。

对于正常相机拍出来的raw图片，通常可以使用rawpy进行直接读取并进行后处理，直接得到全彩色图像。而对于那种没有数据头的纯数据图像，可以使用numpy进行读取，然后用opencv做demosaic得到全彩色图像，然后再另寻办法做awb等后处理操作。关于raw格式图片的读取，通常有raw8，raw10，raw12，raw14，raw16这些格式。其实就是每一个像素值占多少位。而这里存在一些问题，比如用numpy进行文件读取的时候只有uint8和uint16，对于其他格式没法直接读取。我自己测试的结果是12位可以用uint16直接读，但是10位就会出问题。虽然raw的位数有很多种，但是处理起来其实很简单。最简单的处理方法就是直接扩大位数，以raw8转raw10为例，只需要对所有数据*4（也就是2的2次方）即可。而10位转8位只需要/4就可以。这样可以将图片全都转到raw8来处理，当然其他格式之间也可以互相转。raw8是最常用的，因为opencv显示图片之类的方法默认只支持8位图像。其实这个操作我觉得很像normalize之后再还原回去。

**有一个坑就是如果使用m1版的conda是装不了rawpy的，pip会提示找不到。这不是你的问题，我扫了一眼github貌似是不推荐在m1版上使用。**

关于使用numpy+opencv处理图像以及raw格式转换附上一个参考代码

```python
import numpy as np
def raw_convert(raw,source,target):
    return raw*np.power(2,float(target-source)).astype(np.uint16)
def read_raw(file_name,shape):
    # 从raw文件中读取数据
    data = np.fromfile(file,dtype="uint16")
    data.resize(shape)
    return data
def raw2png(raw,path)
    dst=cv2.cvtColor(raw.astype("uint8"), cv2.COLOR_BayerBG2BGR)#我之前看文档，这一步应该直接就做demosaic了
    cv2.imwrite(str(path), dst)
```

## Bayer格式之间互转
参考这个库https://github.com/Jiaming-Liu/BayerUnifyAug
**由于算法的关系（具体可以看仓库指向的论文）Bayer互转后分辨率会稍有变化，当你在numpy resize的时候格式不再是原来分辨率。可以在转完之后直接print一下现在的shape