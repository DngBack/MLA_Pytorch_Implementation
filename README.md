# MLA_Pytorch_Implementation

Multi-head Latent Attention in Deepseekv2.

Paper Link: [DeepseekV2](https://arxiv.org/pdf/2405.04434)

## Overview

Multi-head Latent Attention (MLA) is a variant of multi-head attention which was introduced in the DeepSeek-V2 paper. There are several variants of multi-head attention whose purpose is primarily to reduce the KV-cache size, which is a memory bottleneck that emerges from scaling large models. These methods, which include Group-Query Attention and Multi-Query Attention, are primarily considered /performance tradeoffs/, i.e. the performance is worse, but you get to scale them much further by reducing the memory overhead.

## To do List

- Implementation Standard Multi-head Atention

  - [x] With Defaut
  - [x] With Standard RoPE
  - [x] With Decouple RoPE

- **Implementation Standard Multi-head Latent Atention**

  - [x] With Defaut
  - [x] With Standard RoPE
  - [x] With Decouple RoPE

- Training Script
