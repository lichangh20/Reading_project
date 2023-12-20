# [IMPLICIT CHAIN OF THOUGHT REASONING VIA KNOWLEDGE DISTILLATION](https://arxiv.org/pdf/2311.01460.pdf)

亮点：定义了一个新范式，直接使用模型的hidden states进行implicit reasoning，不需要传统方式的chain-of-thought方法，却可以解决一些之前必须用CoT才能解决的问题，同时速度也提升了很多

主要思想：使用一个emulator model去预测teacher model内部的hidden state的状态，再训练一个Mind-Reading model在已知teacher model内部hidden state的前提下预测最终生成的答案。最终将这两个模型结合，端到端优化，使得student model可以有和teacher model完全不一样的方法

### Mind-Reading Model

假设有L层hidden layer，Intermediate token是T个。在Transformer架构下，每个token在每个hidden layer层都有对应的vector表示。此时transformer架构中总共有[L,T]个Vector

如果L=T，那么直接选取上面矩阵的对角元Vector

否则依旧每个hidden layer选一个，但token间隔$\lceil(\frac{T-1}{L-1})\rceil$这么多

使用Teacher model中select出的vector替换Mind-Reading Model中对应的hidden states，并训练其预测最终输出

### Emulator Model

由于在训练的时候Student Model无法得知Teacher Model中对应的Vector，因此需要对应的Emulator Model来预测Vector具体是什么

问题：模型可能会有多种思考路径，最后优化的结果可能是多种路径的均值，结果跟一个都对不上。解决方案是通过一个全概率公式，计算不同reasoning path下对应的概率

此时又会出现新的问题，model collapsing，即最后多种reasoning path会塌缩到只有几种的情形。解决方案是在loss中让对应位置的token和该处的reasoning path尽量一致(从而增大缺失一种reasoning path带来的损失)。	





补充——知识蒸馏：

1. Softmax中T的含义：
   $q_i=\frac{exp(z_i/T)}{\sum_{j}exp(z_j/T)}$，容易看出随着T的增加$q_i$之间的差距也逐渐变小。形象的称之为`soft`，相应的没有T的Softmax称为`hard`
2. 知识蒸馏是小模型一方面和 teacher model 进行一个`hard`的Softmax，另一方面和 ground truth 进行`soft`的Softmax。Motivation是 teacher model 是大模型，表征空间更大，因此小模型更容易习得大模型的 knowledge，从而更易用更少的数据进行高效训练