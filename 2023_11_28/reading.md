# [DRESS : Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback](https://arxiv.org/pdf/2311.10081.pdf)

### 主要贡献

提出了一种将视觉语言模型与人类对齐的新范式  ，使用NLF(Natural Language Feedback)

之前问题：

- 只依赖微调和人类对齐，没有额外的反馈
- 大部分微调是依靠多轮对话进行的，但对话间的联系和依赖比较弱

本作贡献：

- critique：辨别回答的长处和弱点
- refinement：提供具体提升的建议
- NLF的本质是不可微分的，generalize conditional reinforcement learning来进行训练

总之就是给出的反馈包括一个numerical score与NLF，通过conditional 的交叉熵使得NLF可以对模型的更新起到一定的效果(同时可以利用多轮反馈的结果，加强多轮对话之间的联系)

