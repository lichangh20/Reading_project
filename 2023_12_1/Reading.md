# [A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future](https://arxiv.org/pdf/2309.15402.pdf)

一篇关于CoT的综述，讲了目前CoT发展各个方向的前景，以及之后可以做些什么。[知乎](https://zhuanlan.zhihu.com/p/664263382)有对应的中文讲解，我主要看的知乎讲解，一些和我的project比较match的地方看的原文。Project一个聚焦点在于Self-Refine，参考了以下几篇文章



# [SELF-REFINE: Iterative Refinement with Self-Feedback](https://arxiv.org/pdf/2303.17651.pdf)

感觉是见过最水的文章之一。。。

就是用ChatGPT、GPT3.5、GPT4三个大模型，给一个prompt，自己生成一个response，自己再来使用Natural language 进行 refine。如此是一个epoch，不断迭代，直到达到某个stop point，作者说明这三个大模型在refine之后效果都变得更好。

消融实验发现除了这三个大模型之外这个方法都行不太通，因此 limitation 就是成本高，且适用的模型都是闭源的，xs

不过在这篇文章的 Related work 中又找到了几个和当前project可能更像的，尝试拜读一下



# [Learning to summarize from human feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)

![image-20231201142744731](RL.png)

过程如图，属于典型OpenAI式疯狂烧钱、堆人力的工作

本文聚焦于Text Summary领域的工作，不过也具有较强的可拓展性

首先对于一个Blog，让大模型输出很多对应的Summary，从中随机选出两个，并让一个人来判断这两个谁更好一些

接下来利用这些文本来训练，对 reward model 进行更新，直到 reward model 具有不错的能力

最后使用PPO算法来对生成模型的 policy 进行更新(此时基于假设是 reward model 的效果已经不错了)



# [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/pdf/2207.01780.pdf)

主要针对的是如何根据一段文本生成对应代码的过程。大致 framework 见图：

![image-20231201144746945](CodeRL.png)

也是一个基于一定的 Baseline 基础之上添加强化学习的思想，但由于对代码生成任务了解不多，因此细节没有完全看懂，且跟目前的 Project 关系不大



