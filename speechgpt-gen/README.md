# SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation

<a href='https://0nutation.github.io/SpeechGPT-Gen.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2401.13527'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 

<p align="center">
    <img src="imgs/coi.png" width="60%"> <br>
</p>

## Introduction
Benefiting from effective speech modeling, current Speech Large Language Models (SLLMs) have demonstrated exceptional capabilities in in-context speech generation and efficient generalization to unseen speakers. 
However, the prevailing information modeling process is encumbered by certain redundancies, leading to inefficiencies in speech generation.
We propose Chain-of-Information Generation (CoIG), a method for decoupling semantic and perceptual information in large-scale speech generation. Building on this, we develop SpeechGPT-Gen, an 8-billion-parameter SLLM efficient in semantic and perceptual information modeling. It comprises an autoregressive model based on LLM for semantic information modeling and a non-autoregressive model employing flow matching for perceptual information modeling. Additionally, we introduce the novel approach of infusing semantic information into the prior distribution to enhance the efficiency of flow matching. 
Extensive experimental results demonstrate that SpeechGPT-Gen markedly excels in zero-shot text-to-speech, zero-shot voice conversion, and speech-to-speech dialogue, underscoring CoIG's remarkable proficiency in capturing and modeling speech's semantic and perceptual dimensions.

<p align="center">
    <img src="imgs/model.png" width="95%"> <br>
    Illustration of SpeechGPT-Gen.
</p>


## Code 
We will soon open-source our codes and models, stay tuned!



## Citation
