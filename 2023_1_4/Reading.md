> hys学长推荐的两篇相关文章，读了abstract之后感觉确实很相近，在此记录一下

# [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://arxiv.org/pdf/2306.01693.pdf)

传统的RLHF方法在长文本的反馈下传达信息有限，无法精确定位错误出在哪方面。作者提出Fine-Grained RLHF，在两方面有提升：1) density：每个句子都给reward 2) multiple rewards：训练多个Discriminator，每个都专门评判某一方面的能力

之前的RLHF中，如果两个回答都有multiple undesired behaviors，很难评价孰优孰劣

在long-QA 类型的问题中，作者构建了multiple reward models，并且通过不同的权重加权来结合multiple reward models，最终控制模型的训练过程

### 效果检测

毒性检测任务：在该类任务上toxicity是唯一在意的特性，通过调用标准的Perspective API能够获得一个0-1直接的数值，代表toxicity的大小。结果表明，使用sentence-level的reward效果会比sample-level收敛更快，且更加fluent

原因推测：Sentence-level的给reward可以更方便定为问题出在哪里，代价是每个sample需要调用更多次reward model的query

Long-form QA的任务：

`UW真的是土豪啊！`，作者聘请workers基于ASQA构建了一个高质量的fine-grained Dataset，分为三种错误：C1: irrelevance, repetition, or incoherence ; C2: incorrect or unverifiable facts ; and C3: incomplete information，前两种是sentence-level的，最后一种是sample-level的，并且让workers重新构建了一个能解决所有标注问题的好版本答案

对于C1,C2,C3三种错误，分别构建三个Reward Model，focus on sub-sentence, sentence, full sequence level

对于上述三种错误分别定义了相应的Loss；与此同时Preference-based model也被作者用于进行对比实验

定制实验：由于Loss是上述三个Loss的加权，作者发现通过分配不同的权重，模型的输出会向不同的方向倾斜

消融实验：作者发现减少一个Reward Model，模型在相应方面的能力有所下降；同时检测了Reward Model自身打分的能力；并将微调之后的模型与ChatGPT模型进行了对比，展现出了更强大的性能

有了reward之后，通过PPO算法来对目标进行优化

目前的重要小问题：

1. 怎么结合多个reward model的score的？没有太看懂
2. KL散度这里面是干什么的？很多地方都会用到

目前不是很重要的小问题：

1. PPO算法，语言生成as a MDP简单了解一下