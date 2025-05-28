# 12-th Place Solution for Kaggle "Drawing with LLMs" Competition

This is the training code for Kaggle "[Drawing with LLMs](https://www.kaggle.com/competitions/drawing-with-llms/)" competition.

The original solution can be find in [this discussion](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034).

This repo is based on the fantastic work: [Flow-GRPO](https://github.com/yifan123/flow_grpo).

## Installation

You can create the environment for Flow-GRPO with following command:

```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```

## Usage
Single-node training:
```bash
bash scripts/single_node/main.sh
```