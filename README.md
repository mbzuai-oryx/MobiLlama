# 📱🦙 MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT



<p align="center">
    <img src="./images/MobileLLaMa.png" height="400px" alt="Oryx MobiLLama">
</p>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx MobiLLama">
</p>

<p align="center">
   <a href="https://github.com/mbzuai-oryx/MobiLlama/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"></a>
</p>

#### [Omkar Thawakar](https://scholar.google.com/citations?user=jLNKLsgAAAAJ&hl=en&oi=ao), [Ashmal Vayani](https://scholar.google.com/citations?user=LJWxVpUAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/), [Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ), [Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?user=lkWfR08AAAAJ&hl=en), [Timothy  Baldwin](https://scholar.google.com/citations?user=wjBD1dkAAAAJ&hl=en), [Eric Xing](https://scholar.google.com/citations?user=5pKTRxEAAAAJ&hl=en) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)

#### **Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2402.14818)
🤗 [![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)](https://huggingface.co/MBZUAI/MobiLlama)
🤗 <a href="https://huggingface.co/datasets/LLM360/AmberDatasets">[Pretraining Dataset Download]</a> 

---

## 📢 Latest Updates
- **Feb-26-24**- Arxiv Preprint is released!
- **Feb-25-24**- Code (Training and Evaluation scripts) is released!


## Overview

`Bigger the better` has been the predominant trend in recent Large Language Models (LLMs) development.
However, LLMs do not suit well for scenarios that require on-device processing, energy efficiency, low memory footprint, and response efficiency. These requisites are crucial for privacy, security, and sustainable deployment. 
This paper explores the `less is more` paradigm by addressing the challenge of designing accurate yet efficient Small Language Models (SLMs) for resource constrained devices. 
Our primary contribution is the introduction of an accurate and fully transparent open-source 0.5 billion (0.5B) parameter SLM, named `MobiLlama`, catering to the specific needs of resource-constrained computing with an emphasis on enhanced performance with reduced resource demands.
`MobiLlama` is a SLM design that initiates from a larger model and applies a careful parameter sharing scheme to reduce both the pre-training and the deployment cost.

## ⚡ Model Download
             
| Model Name           | Link Download                                  |
|-----------------------------------------------------|----------------------------------------------------------------------|
| MobiLlama-05B           | [HuggingFace](https://huggingface.co/MBZUAI/MobiLlama-05B)  |
| MobiLlama-08B           | [HuggingFace](https://huggingface.co/MBZUAI/MobiLlama-08B) |
| MobiLlama-1B            | [HuggingFace](https://huggingface.co/MBZUAI/MobiLlama-1B) |
| MobiLlama-05B-Chat      | [HuggingFace](https://huggingface.co/MBZUAI/MobiLlama-05B-Chat) |
| MobiLlama-1B-Chat       | [HuggingFace](https://huggingface.co/MBZUAI/MobiLlama-1B-Chat) |

## Model Description

- **Model type:** Language model designed using the architecture of LLaMA-7B
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Resources for more information:**
  - [Training Code](https://github.com/mbzuai-oryx/MobiLlama)
  - [Data Preparation](https://github.com/LLM360/amber-data-prep)
  - [Metrics]()
  - [Fully processed Amber pretraining data](https://huggingface.co/datasets/LLM360/AmberDatasets)


# Loading MobiLlama 

```python
from .model_utils.modeling_mobillama import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("MBZUAI/MobiLlama-05B")
model = LlamaForCausalLM.from_pretrained("MBZUAI/MobiLlama-05B")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## Dataset

Download the preprocessed Amber data from [huggingface](https://huggingface.co/datasets/LLM360/AmberDatasets). The entire training data has 360 chunks totalling the size of ~8 TB. Amber dataset contains total 1.2 Trillion tokens with gathered from different data sources shown below.

| Subset      | Tokens (Billion) |
| ----------- | ----------- |
| Arxiv      | 30.00       |
| Book   | 28.86        |
| C4   | 197.67        |
| Refined-Web   | 665.01        |
| StarCoder   | 291.92        |
| StackExchange   | 21.75        |
| Wikipedia   | 23.90        |
| Total | 1259.13 | 

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install from source (recommended for training/fine-tuning) run:

```bash
conda create -n mobillama python=3.10
conda activate mibillama
git clone https://github.com/mbzuai-oryx/MobiLlama.git
cd MobiLlama
pip install -r  requirements.txt
```

## pretrain
For MobiLlama (using 20 nodes of A100 80GB GPUS)
```bash
sbatch pretrain.sh
```
For `large-base` use main_largebase.py in L:11 of pretrain.sh

## 🔎 Evaluation

We used [Analysis-360](https://github.com/LLM360/Analysis360) to evaluate our model on different llm benchmarks. 



## 📊  Results

| Model Name         | #Params | HellaSwag | Truthfulqa | MMLU  | Arc_C | CrowsPairs | piqa  | race  | siqa  | winogrande | Average |
|--------------------|---------|-----------|------------|-------|-------|------------|-------|-------|-------|------------|---------|
| gpt-neo-125m       | 0.15B   | 30.26     | 45.58      | 25.97 | 22.95 | 61.55      | 62.46 | 27.56 | 40.33 | 51.78      | 40.93   |
| tiny-starcoder     | 0.17B   | 28.17     | 47.68      | 26.79 | 20.99 | 49.68      | 52.55 | 25.45 | 38.28 | 51.22      | 37.86   |
| cerebras-gpt-256m  | 0.26B   | 28.99     | 45.98      | 26.83 | 22.01 | 60.52      | 61.42 | 27.46 | 40.53 | 52.49      | 40.69   |
| opt-350m           | 0.35B   | 36.73     | 40.83      | 26.02 | 23.55 | 64.12      | 64.74 | 29.85 | 41.55 | 52.64      | 42.22   |
| megatron-gpt2-345m | 0.38B   | 39.18     | 41.51      | 24.32 | 24.23 | 64.82      | 66.87 | 31.19 | 40.28 | 52.96      | 42.81   |
| LiteLlama          | 0.46B   | 38.47     | 41.59      | 26.17 | 24.91 | 62.90      | 67.73 | 28.42 | 40.27 | 49.88      | 42.26   |
| gpt-sw3-356m       | 0.47B   | 37.05     | 42.55      | 25.93 | 23.63 | 61.59      | 64.85 | 32.15 | 41.56 | 53.04      | 42.48   |
| pythia-410m        | 0.51B   | 40.85     | 41.22      | 27.25 | 26.19 | 64.20      | 67.19 | 30.71 | 41.40 | 53.12      | 43.57   |
| xglm-564m          | 0.56B   | 34.64     | 40.43      | 25.18 | 24.57 | 62.25      | 64.85 | 29.28 | 42.68 | 53.03      | 41.87   |
| Lamini-GPT-LM      | 0.59B   | 31.55     | 40.72      | 25.53 | 24.23 | 63.09      | 63.87 | 29.95 | 40.78 | 47.75      | 40.83   |
| **MobiLlama (Ours)** | **0.5B**   | **52.52**    | **38.05**     | **26.45**| **29.52**| **64.03**     | **72.03**| **33.68**| **40.22**| **57.53**     | **46.00**  |
| Lamini-GPT-LM      | 0.77B   | 43.83     | 40.25      | 26.24 | 27.55 | 66.12      | 69.31 | 37.12 | 42.47 | 56.59      | 45.49   |
| **MobiLlama (Ours)** | **0.8B**   | **54.09**    | **38.48**     | **26.92**    | **30.20** | **64.82** | **73.17** | **33.37** | **41.60** | **57.45** | **46.67** |  

`The table provides a comparative analysis of various models, including our MobiLlama, across several LLM benchmarks. It highlights MobiLlama's superior performance, particularly in its 0.5B and 0.8B configurations, showcasing its efficiency and effectiveness in processing complex language tasks. This comparison underscores MobiLlama's advancements in achieving higher accuracy and demonstrates its potential as a leading solution in the field of LLM.`

---

| Model         | #Params | HellaSwag | Truthfulqa | MMLU | Arc_C | CrowsPairs | piqa | race | siqa | winogrande | Average |
|---------------|---------|-----------|------------|------|-------|------------|------|------|------|------------|---------|
| Boomer        | 1B      | 31.62     | 39.42      | 25.42| 22.26 | 61.26      | 57.99| 28.99| 40.32| 50.98      | 39.80   |
| Pythia-Dedup  | 1B      | 49.63     | 38.92      | 24.29| 29.09 | 67.11      | 70.23| 32.44| 42.63| 53.98      | 45.36   |
| Falcon-RW     | 1B      | 63.12     | 35.96      | 25.36| 35.06 | 69.04      | 74.10| 36.07| 40.23| 61.88      | 48.98   |
| TinyLlama     | 1.1B    | 60.22     | 37.59      | 26.11| 33.61 | 70.60      | 73.28| 36.45| 41.65| 59.18      | 48.74   |
| OLMo          | 1.2B    | 62.50     | 32.94      | 25.86| 34.45 | 69.59      | 73.70| 36.74| 41.14| 58.90      | 48.42   |
| Cerebras-GPT  | 1.3B    | 38.51     | 42.70      | 26.66| 26.10 | 63.67      | 66.75| 30.33| 42.42| 53.59      | 43.41   |
| Lamini        | 1.3B    | 38.05     | 36.43      | 28.47| 26.62 | 64.62      | 67.89| 33.39| 43.19| 50.59      | 43.25   |
| OPT           | 1.3B    | 54.50     | 38.67      | 24.63| 29.60 | 70.70      | 72.47| 34.16| 42.47| 59.74      | 47.43   |
| GPT-NEO       | 1.3B    | 48.49     | 39.61      | 24.82| 31.31 | 65.67      | 71.05| 34.06| 41.81| 57.06      | 45.98   |
| Pythia-Deduped| 1.4B    | 55.00     | 38.63      | 25.45| 32.59 | 67.33      | 72.68| 34.64| 42.68| 56.90      | 47.32   |
| **large-base**| **1.2B**| **62.99** | **35.90**  | **24.79**| **34.55** | **68.49**  | **75.57**| **35.31**| **41.96**| **62.03**  | **49.06**  |

`Comprehensive comparisons with existing < 2B params fully open-source LLM models on 9 benchmarks. Our 1.2B "large-base" model pre-trained on 1.2T tokens achieves superior performance compared to both the recent OLMo 1.17B model and TinyLlama 1.1B model, which are pre-trained on a substantially larger data of 3T tokens.`

## 📱 MobiLlama on Android

To run our  model on an android app, please download  and install the APK from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/omkar_thawakar_mbzuai_ac_ae/EhRfGdmgFVVNvIRfy1EgLwEBjbk_eg3UmNg_zjz7PMTsmg?e=NBuJo8). 

## 🙏 Acknowledgements

+ We thank [LLM-360](https://github.com/LLM360/amber-train) for fully transparent and open-source implementation of their language model. MobiLlama repo is built using [LLM-360](https://github.com/LLM360/amber-train). 


## 📜 Citation
```bibtex
coming soon !
```