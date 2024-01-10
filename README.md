<div align="center">
  
# 【ICASSP'2024 🎄】WAVER: Writing-Style Agnostic Text-Video Retrieval Via Distilling Vision-Language Models Through Open-Vocabulary Knowledge
  
[![Conference](https://img.shields.io/badge/ICASSP-2024-FGD94D.svg)](https://2024.ieeeicassp.org/)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2312.09507-FF6B6B.svg)](https://arxiv.org/abs/2312.09507)
</div>

The implementation of ICASSP 2024 paper [WAVER: Writing-Style Agnostic Text-Video Retrieval Via Distilling Vision-Language Models Through Open-Vocabulary Knowledge](https://arxiv.org/abs/2312.09507)

## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{le2024waver,
  title={WAVER: Writing-style Agnostic Text-Video Retrieval via Distilling Vision-Language Models Through Open-Vocabulary Knowledge},
  author={Le, Huy and Kieu, Tung and Le, Ngan},
  booktitle={ICASSP},
  pages={1--5},
  year={2024},
}
```

## 📕 Overview
Text-video retrieval, a prominent sub-field within the domain of multimodal information retrieval, has witnessed remarkable growth in recent years. However, existing methods assume video scenes are consistent with unbiased descriptions. These limitations fail to align with real-world scenarios since descriptions can be influenced by annotator biases, diverse writing styles, and varying textual perspectives. To overcome the aforementioned problems, we introduce WAVER, a cross-domain knowledge distillation framework via vision-language models through open-vocabulary knowledge designed to tackle the challenge of handling different writing styles in video descriptions. WAVER capitalizes on the open-vocabulary properties that lie in pre-trained vision-language models and employs an implicit knowledge distillation approach to transfer text-based knowledge from a teacher model to a vision-based student. Empirical studies conducted across four standard benchmark datasets, encompassing various settings, provide compelling evidence that WAVER can achieve state-of-the-art performance in text-video retrieval task while handling writing-style variations.

### Setup code environment
```shell
conda create -n video_retrieval python=3.9
conda activate video_retrieval
pip install -r requirements.txt
```

### Download CLIP Model

```shell
cd tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

### Download Data
```shell
cd data/{dataset}
Download Link [Here]{https://github.com/Fsoft-AIC/WAVER/releases/tag/v0.0.1}
```

###  Train on MSR-VTT
```shell
python -m torch.distributed.run \
--master_port 2502 \
--nnodes=1 \
--nproc_per_node=2 \
main.py \
--do_train 1 \
--workers 6 \
--n_display 20 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--base_encoder ViT-B/32
--agg_module seqTransf \
--top_k 1 \
--interaction wti \
--wti_arch 2 \
--drop_last \
--prompt_tuning \
--output_dir ${OUTPUT_PATH}
```

## 🎗️ Acknowledgments
* This code implementation are adopted from [CLIP](https://github.com/openai/CLIP) and [DRL](https://github.com/foolwood/DRL).
We sincerely appreciate for their contributions.
