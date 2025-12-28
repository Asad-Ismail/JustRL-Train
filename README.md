# JustRL-LLM-Trianing
Benchmarking RL trainings

1. Implemented training for JustRL paper [github](https://github.com/thunlp/JustRL), [paper](https://arxiv.org/abs/2512.16649)

Ti run training use 

export PYTORCH_ALLOC_CONF="expandable_segments:True"
accelerate launch train_justRL.py
