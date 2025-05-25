# Codebase for [Preprint](https://arxiv.org/abs/2502.04354)

## "Reviving The Classics: Active Reward Modeling in Large Language Model Alignment"

#### Authors: Yunyi Shen*, Hao Sun*, Jean-Francois Ton. The first two authors contribute equally.
Accepted at ICML'25.

[ [Preprint](https://arxiv.org/abs/2502.04354) ]       |       [[Embeddings](https://github.com/holarissun/embedding-based-llm-alignment)]]

_We have a series of work focusing on reward models in RLHF:_
- Part I. Reward Model Foundation [ICLR'2025 Oral](https://arxiv.org/abs/2411.04991), [Code Repo](https://github.com/holarissun/RewardModelingBeyondBradleyTerry)
- Part II. Active Reward Modeling (This repo) [ICML 2025](https://arxiv.org/abs/2502.04354)
- Part III. Accelerating Reward Model Research with our Infra. [Preprint](https://arxiv.org/abs/2502.04357), [Code Repo](https://github.com/holarissun/embedding-based-llm-alignment)

## Structure of the repo
Algorithms we tested were implemented in `model`, there are two algorithms from other authors, namely [coreset (Huggins et al. 2016)](https://proceedings.neurips.cc/paper/2016/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html) in `lrcoresets` and [batchBALD (Kirsch et al 2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html) in `batchbald_redux`, we did minimal modification to make sure then can be compitable with our computation environment.

Experiment code to be released soon after we remove unnecessary parts due to our specific computation environment.

