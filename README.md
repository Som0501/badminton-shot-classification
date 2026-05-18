# Badminton Shot Classification with Swin Transformer

Fine-grained classification of **18 badminton shot types** from match video frames, using a Swin Transformer backbone. This was my MSc-level dissertation project at the University of Surrey (PG Dip AI, 2024–2025).

> **TL;DR** — Swin-Tiny classifier trained on a curated 18-class video dataset. Frame-level baseline reaching **79.21% Top-3** and **92.22% Top-5** accuracy on the held-out test set, with class-imbalance handling and modern training tricks throughout.

---

## Problem

Fine-grained sports-action recognition is hard for three reasons:

1. **Many visually similar classes.** Different badminton shots (e.g. *short serve* vs *long serve*, *forehand drive* vs *backhand drive*) differ by subtle pose and racket trajectory cues.
2. **Severe class imbalance.** Common shots (*clear*, *drop*) dominate the dataset; rare shots have far fewer examples.
3. **Limited public datasets** of labelled badminton shots compared to general action-recognition benchmarks.

This project builds a frame-level baseline using a modern transformer-based vision backbone, with the architecture and training pipeline designed to extend cleanly to multi-frame temporal models in future work.

---

## Approach

### Model
- **Backbone:** [`swin_tiny_patch4_window7_224`](https://arxiv.org/abs/2103.14030) (Swin Transformer-Tiny), pre-trained on ImageNet-1k, loaded via `timm`.
- **Head:** Linear classifier over 18 classes, with dropout (p=0.3) for regularisation.
- **Input:** 224×224 RGB frame sampled from each video.

### Data pipeline
- **Frame extraction:** Single representative frame per video via OpenCV (`cv2.VideoCapture`).
- **Splits:** 80 / 10 / 10 train / validation / test (random split, stratified by overall distribution).
- **Augmentation (train):** Random horizontal flip, ±15° rotation, colour jitter (brightness/contrast/saturation/hue), and random erasing.
- **Normalisation:** ImageNet mean/std.

### Handling class imbalance
Two complementary mechanisms:
- **`WeightedRandomSampler`** during training, with weights inversely proportional to class frequency.
- **Class-weighted `CrossEntropyLoss`** using `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`.

### Training
- **Framework:** PyTorch Lightning 2.x.
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5).
- **Scheduler:** OneCycleLR (per-step).
- **Precision:** Mixed-precision (`16-mixed`) for A100 throughput.
- **Epochs:** Up to 50 with `ModelCheckpoint` on `val_loss` (top-3 kept).
- **Logging:** TensorBoard (Weights & Biases used in earlier iterations).

---

## Results

Evaluated on the held-out test split (10% of the dataset).

| Metric          | Score   |
|-----------------|---------|
| Top-1 accuracy  | *(see report)* |
| **Top-3 accuracy**  | **79.21%** |
| **Top-5 accuracy**  | **92.22%** |

The large Top-5 vs Top-1 gap is consistent with the fine-grained-classification literature: the model correctly identifies the *family* of shot (e.g. "drive") but confuses sub-classes (forehand vs backhand drive) that differ by player pose alone. Per-class accuracy and full confusion-matrix analysis are in the notebook.

---

## Repo contents

```
.
├── DISS_FINAL.ipynb     # Full training + evaluation pipeline
├── README.md            # You are here
├── LICENSE              # MIT
└── .gitignore           # Python
```

The notebook is self-contained: data loading → augmentation → model → training → evaluation → confusion matrix → per-class accuracy table.

---

## How to run

The notebook was developed and trained on **Google Colab with an A100 GPU**.

1. **Open the notebook in Colab** (or any Jupyter environment with GPU).
2. **Mount your dataset.** The notebook expects a directory of the form:
   ```
   <root>/
     ├── ClassName1/
     │     ├── clip001.mp4
     │     └── ...
     ├── ClassName2/
     │     └── ...
     └── ...
   ```
   Update `drive_path` in Cell 4 to point at your data root.
3. **Install dependencies** (Cell 1):
   ```bash
   pip install timm pytorch-lightning wandb opencv-python scikit-learn seaborn
   ```
4. **Run all cells.** Training defaults: 50 epochs, batch size 32, mixed precision.

> **Note on dataset:** the source video dataset is not redistributed in this repo. Any 18-class video dataset following the directory structure above will work as a drop-in.

---

## Honest limitations

I'm flagging these explicitly because they matter:

- **Frame-level, not temporal.** The current pipeline uses one frame per video. True spatio-temporal modelling (multiple frames + temporal attention) is the natural next step and would likely close the Top-1 vs Top-5 gap.
- **First-frame sampling.** I take the first readable frame per clip rather than a key-action frame or a sampled set. A keyframe-detection step (e.g. picking frames near maximum optical-flow magnitude) is a low-effort improvement.
- **Pre-trained ImageNet backbone.** Sports-action distribution differs from ImageNet; pre-training or fine-tuning on a larger action dataset (Kinetics-400) before this task would likely lift performance.
- **Single backbone evaluated.** ConvNeXt, ViT, and dedicated video backbones (TimeSformer, VideoMAE, X3D) were not directly compared in this run.

---

## Future work

1. **Temporal models** — TimeSformer, VideoMAE, or SlowFast to use multiple frames per clip.
2. **Frame-sampling strategy** — optical-flow-based keyframe selection instead of first-frame.
3. **Targeted augmentation** for confusable class pairs (forehand vs backhand) using pose-aware mixing.
4. **Backbone comparison** — ViT-B/16, ConvNeXt-T, EfficientNet-V2 against Swin under matched training budgets.

---

## Tech stack

`Python` · `PyTorch` · `PyTorch Lightning` · `timm` · `OpenCV` · `scikit-learn` · `Seaborn` · `Matplotlib` · `Weights & Biases` · `TensorBoard`

---

## About

This was my dissertation project for the **Postgraduate Diploma in Artificial Intelligence at the University of Surrey** (Guildford, UK, 2024–2025).

📫 [LinkedIn](https://www.linkedin.com/in/som-kapoor-44a012233?)· [Email](mailto:somkapoor0501@gmail.com)

---

## License

MIT — see [LICENSE](./LICENSE).
