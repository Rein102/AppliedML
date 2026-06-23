## List of available backbone models you can use

### Vision Transformer models

model_name: ViT-B-32,
pretrained: laion2b_s34b_b79k,
description: Smallest and fastest model, good for quick tests but lower accuracy.

model_name: ViT-B-16,
pretrained: laion2b_s34b_b88k,
description: Small model with finer 16x16 patches, better detail than ViT-B-32 at similar speed.

model_name: ViT-B-16-plus-240,
pretrained: laion400m_e31,
description: Default model in FADE, trained at 240px resolution, solid balance of speed and accuracy.

model_name: ViT-L-14,
pretrained: openai,
description: OpenAI's original large CLIP model, strong general-purpose features.

model_name: ViT-L-14,
pretrained: laion2b_s32b_b82k,
description: Same architecture as above but trained on larger LAION-2B dataset, generally stronger.

model_name: ViT-L-14-336,
pretrained: openai,
description: ViT-L trained at higher 336px resolution, better for fine-grained anomaly detection.

model_name: ViT-g-14,
pretrained: laion2b_s34b_b88k,
description: Large model with strong representations, currently set in run.sh.

model_name: ViT-H-14,
pretrained: laion2b_s32b_b79k,
description: Very large model, high accuracy but requires significant GPU memory.

model_name: ViT-bigG-14,
pretrained: laion2b_s39b_b160k,
description: Largest available model, best accuracy but very slow and memory-hungry.

### ResNet models

model_name: RN50,
pretrained: openai,
description: Classic ResNet-50 CLIP model from OpenAI, fast but weakest feature quality.

model_name: RN101,
pretrained: openai,
description: Deeper ResNet-101, slightly stronger than RN50 at a small speed cost.]

model_name: RN50x16,
pretrained: openai,
description: Wide ResNet scaled 16x, much stronger than RN50 but significantly slower.

model_name: RN50x64,
pretrained: openai,
description: Largest ResNet variant from OpenAI, high accuracy but very memory intensive.

### ConvNeXT models

model_name: convnext_base_w,
pretrained: laion2b_s13b_b82k,
description: Modern CNN backbone, good balance of speed and accuracy, trained on LAION-2B.

model_name: convnext_large_d,
pretrained: laion2b_s26b_b102k_augreg,
description: Larger ConvNeXT with augmented regularisation, strong CNN alternative to ViT-L.

model_name: convnext_xxlarge,
pretrained: laion2b_s34b_b82k_augreg,
description: Largest ConvNeXT model, competitive with large ViTs for anomaly detection.

### EVA models

model_name: EVA02-B-16,
pretrained: merged2b_s8b_b131k,
description: EVA-02 small model, strong vision-language features trained on merged datasets.

model_name: EVA02-L-14,
pretrained: merged2b_s4b_b131k,
description: EVA-02 large model, excellent feature quality and often outperforms ViT-L.

model_name: EVA02-E-14,
pretrained: laion2b_s4b_b115k,
description: Largest EVA-02 model, state-of-the-art features but requires significant GPU memory.

model_name: EVA01-g-14,
pretrained: laion400m_s11b_b41k,
description: First generation EVA giant model trained on LAION-400M.