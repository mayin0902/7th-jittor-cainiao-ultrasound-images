# 7th Jittor Cainiao Ultrasound Image Classification

![Jittor](https://img.shields.io/badge/Framework-Jittor-blue)
![Rank](https://img.shields.io/badge/National%20Rank-7%20%2F%20Finalist-green)
![Task](https://img.shields.io/badge/Task-Breast%20Ultrasound%20Classification-purple)

**National Rank 7 solution** for Track 1 of the **Jittor AI Algorithm Challenge**: intelligent screening and grading of breast ultrasound images.

The Jittor AI Algorithm Challenge was founded in 2021 under the guidance of the Department of Information Sciences of the National Natural Science Foundation of China, organized by the Beijing National Research Center for Information Science and Technology and the Tsinghua University - Tencent Joint Laboratory for Internet Innovation Technology. The competition is based on Tsinghua University's Jittor deep learning framework.

## Result

| Field | Value |
| --- | --- |
| Competition | Jittor AI Algorithm Challenge, Track 1 |
| Chinese Name | 计图(Jittor)人工智能算法挑战赛 |
| Task | Ultrasound image intelligent screening and grading |
| Organizer | Beijing National Research Center for Information Science and Technology; Tsinghua University - Tencent Joint Laboratory for Internet Innovation Technology |
| Guidance | Department of Information Sciences, National Natural Science Foundation of China |
| Framework | Jittor |
| National Rank | **7** |

## Method Overview

- Convert single-channel ultrasound images into three-channel inputs to reuse ImageNet-pretrained visual backbones.
- Use a Jittor implementation of EfficientNetV2-S as the main feature extractor.
- Fuse multi-scale features from the last and penultimate backbone stages to retain both semantic and fine-grained texture information.
- Add a BN-Neck and multi-dropout classification head to reduce overfitting under limited medical data.
- Train with 5-fold cross validation, AdamW, cosine annealing, and carefully bounded image augmentations.
- Support checkpoint ensemble at inference time with configurable model weights.

## Repository Layout

```text
config/
  config.yaml
data/
  dataset.py
  transform.py
models/
  jit_model.py
  jimm/
utils/
  comm.py
  trainer.py
train.py
test.py
test.sh
```

## Quick Start

```bash
python -m pip install -r requirements.txt
python train.py --cfg config/config.yaml --df_dir ./data_jittor/train.csv --data_dir ./data_jittor/images
```

Inference with checkpoint ensembling:

```bash
python test.py \
  --test_dir ../TestSetB \
  --checkpoint ../checkpoint/checkpoint1.pkl ../checkpoint/checkpoint2.pkl ../checkpoint/checkpoint3.pkl \
  --weights 0.4 0.4 0.2 \
  --result_path ./result.txt
```

## Core Model

The model in `models/jit_model.py` extracts two complementary feature levels and averages predictions from multiple dropout heads:

```python
pooled_last = self.base_net.global_pool(x).flatten(1)
pooled_mid = self.base_net.global_pool(second_last_block_output).flatten(1)
h = jt.concat([pooled_mid, pooled_last], dim=1)
logits = sum(self.head(dropout(h)) for dropout in self.dropouts) / len(self.dropouts)
```

This design is lightweight, stable under small-sample training, and keeps the full training and inference pipeline inside the Jittor framework.

## Notes

Raw ultrasound images, checkpoints, prediction outputs, and local cache files are intentionally excluded.
