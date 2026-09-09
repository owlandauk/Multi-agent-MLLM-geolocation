---
title: "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
authors: Alec Radford, Jong Wook Kim, et al. (OpenAI)
venue: ICML
year: 2021
arxiv: 2103.00020
topics: [contrastive pretraining, vision-language, foundation encoder]
base_model: CLIP (ViT / ResNet image encoder + text encoder)
benchmarks: [zero-shot ImageNet, many]
role: FOUNDATION ENCODER underlying the retrieval-geolocation family
---

# CLIP

CLIP learns a joint image-text embedding via contrastive pretraining on web-scale pairs. It is the **foundation encoder** beneath the entire [[retrieval-geolocation-family]].

## Relation to this thesis
- Image encoder for [[geoclip]] (aligned to a GPS encoder), [[pigeon]]/[[streetclip]] (geocell classification), [[georanker]], [[pinpoint]].
- Contrast with the generative-MLLM paradigm ([[qwen25-vl]], [[geobayes]]) the thesis's pure-MLLM line operates in.
