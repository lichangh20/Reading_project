# [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

> 由于我不搞Diffusion，难以理解其中涉及的诸多细节，但这篇工作实在是太过强大与有意思，因此以科普和应用的角度了解一下这篇文章实际应用

五种初级阶段常见模型：

- Openpose_full：抓取人物各个部位的信息，包括手、面部表情等，可以实现动作的控制
- Depth_Leres++：抓取深度图，远的地方是黑色，近的地方白色，可以实现景深的控制(例如两手交叉的形状，Openpose可能无法识别，这时候Depth效果会更好)
- Canny：抓取边缘信息(重要的应用：插画师的线稿图上色)
- Head：抓取柔和边缘，相当于在Canny的基础上减少对AI的控制
- Scribble：在前者的基础上进一步减少限制
- Openpose + Depth：可以实现更加精准的控制，例如更好的识别动作 + 深度信息