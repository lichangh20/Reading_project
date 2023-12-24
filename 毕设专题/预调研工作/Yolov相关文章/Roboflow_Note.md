# Roboflow网站blog阅读笔记

> 在调研YOLO V4 & V5文章时，发现Roboflow网站上的blog讲解的清晰而全面，遂用一个Note专门记录学得的新知识

# [What is YOLOv5? A Guide for Beginners.](https://blog.roboflow.com/yolov5-improvements-and-evaluation/)

文章对YOLO V5用到的技术进行了介绍，并说出了其贡献。

V4和V5的最大贡献：将CV中其他领域的很多突破集成进了YOLO算法中，并证明通过这些集成提升了YOLO目标检测的能力

此外，YOLOV5将C语言编写的DarkNet集成到了Pytorch中，使得科研突破更加容易，这也是很大的贡献

# [Data Augmentation in YOLOv4](https://blog.roboflow.com/yolov4-data-augmentation/)

介绍了YOLO V4中的各种Data Augementation 的 trick，详情见Blog，其中一种很重要的 trick 就是 Mosaic Augmentation

# [Getting Started with Data Augmentation in Computer Vision](https://blog.roboflow.com/boosting-image-detection-performance-with-data-augmentation/)

作者指出对于稀疏的CV测试集而言，想要在不收集新数据的前提下提升性能，Data Augmentation是很好的方法。

简单的图形旋转之类的就可以起到很好的效果

# [What is Image Preprocessing and Augmentation?](https://blog.roboflow.com/why-preprocess-augment/)

- Prepocess：通过对图片进行处理，使得符合要求
  - [Resize](https://blog.roboflow.com/you-might-be-resizing-your-images-incorrectly/)：探讨了对图片Resize的trick，一般从比较小的图片开始训练，之后慢慢变大，可以用小图片训出来的ckpt去initialize大图片的model；还探讨了其他一些Trick
  - [Auto-Orientation](https://blog.roboflow.com/exif-auto-orientation/)：探讨有时候需要调整视角，来确保不同图片的视角相同(例如 x-y 坐标等)
  - [Contrast](https://blog.roboflow.com/when-to-use-contrast-as-a-preprocessing-step/)：有些图片通过调对比度，使得不同pixel间区别更明显，有利于识别物体边界，常见的如文本中的文字识别
- Augmentation：通过加入更多图片，提供更多训练样本

对于输入图片，通常先Preprocess再Augmentation；Preprocess对训练集和测试集都用，Augmentation只对训练集用

# [Responding to the Controversy about YOLOv5](https://blog.roboflow.com/yolov4-versus-yolov5/)

记载了YOLO V4和V5之间一些有趣的新闻

将V4和V5之间进行比对

V4基于DarkNet，C语言编写，使用较复杂，适合追求SOTA的人

V5基于Pytorch，Python编写，上手简单，适合想要使用的人

还对其他各个方面进行全方位比较

# [What is Mean Average Precision (mAP) in Object Detection?](https://blog.roboflow.com/mean-average-precision/)

对CV中一种常见指标——MAP进行了分析说明，主要是对Precision-Recall的一种均衡

CV中考虑bounding box误差的一种方式是IoU，AP通过计算不同IoU threshold的一个面积和得到当前PR曲线的一个整体趋势；mAP通过对不同AP进行求和平均，得到对所有不同class PR曲线的性质。

越高的mAP代表模型有越好的PR整体性能，我们认为其性能更好