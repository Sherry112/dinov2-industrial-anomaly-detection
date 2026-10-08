# Industrial Visual Anomaly Detection with DINOv2

## Project summary

I built an end-to-end one-class visual anomaly-detection system for all 15
categories of the MVTec AD industrial inspection benchmark. The system combines
multi-layer DINOv2 Vision Transformer features, approximate greedy coreset
selection, cosine nearest-neighbor retrieval, and normal-validation decision
thresholds. It detects and localizes defects without using anomalous images for
model fitting or threshold selection.

The implementation is a reproducible Python application rather than a
notebook-only experiment. It supports fitting, evaluation, checkpointing,
single-category runs, full-benchmark orchestration, per-category artifacts,
aggregate CSV/JSON reports, and an interactive Streamlit inference dashboard.

## Business problem

Industrial inspection data is highly imbalanced. Normal products are plentiful,
but real defects are rare, varied, and expensive to label. A supervised
classifier may fail on a defect type absent from its training set.

This one-class approach learns the distribution of normal local patterns.
Regions unlike the normal reference bank receive high anomaly scores, making the
system suitable as a screening or decision-support baseline for novel defects.

## Architecture

```text
Normal train/good images
          │
          ├── deterministic 90% fit split
          │          │
          │          ▼
          │   Frozen DINOv2 blocks 8 + 11
          │          │
          │          ▼
          │   normalized 768-D patch descriptors
          │          │
          │          ▼
          │   projected greedy k-center coreset (2,048 patches)
          │
          └── held-out 10% normal calibration split
                     │
                     ▼
              image and pixel thresholds

Test image → identical DINOv2 features → cosine nearest-neighbor distances
                                             │
                                  ┌──────────┴──────────┐
                                  ▼                     ▼
                           image decision        anomaly heatmap
```

## Technical design

- **Dataset:** all 15 object and texture categories from MVTec AD.
- **Backbone:** pretrained, frozen DINOv2 ViT-S/14.
- **Multi-layer representation:** normalized patch tokens from transformer
  blocks 8 and 11 are concatenated and normalized into 768-dimensional vectors.
- **Spatial resolution:** 256 patch descriptors per 224×224 image (16×16 grid).
- **Validation:** deterministic 90/10 split of normal training images per
  category; test data is excluded from fitting and calibration.
- **Coreset:** a seeded candidate pool is projected to 32 dimensions; greedy
  farthest-first traversal retains 2,048 representative patches while the full
  768-dimensional vectors are stored.
- **Patch score:** cosine distance from the closest normal coreset embedding.
- **Image score:** mean of the highest-scoring 1% of image patches.
- **Image threshold:** finite-sample-corrected upper normal-score order statistic
  targeting a 5% normal-image false-positive rate.
- **Pixel threshold:** empirical 99th percentile of held-out normal pixel scores.
- **Evaluation:** AUROC, average precision, accuracy, precision, recall,
  specificity, F1, and confusion counts at the preselected thresholds.

The detector is non-parametric: DINOv2 is not fine-tuned, and its learned state
consists of a category-specific normal memory bank plus calibrated thresholds.

## Full benchmark results

The pipeline completed all 15 categories with zero failures using one fixed
configuration and seed.

| Macro-average metric | Result |
|---|---:|
| Image AUROC | **0.9716** |
| Image average precision | **0.9853** |
| Pixel AUROC | **0.9647** |
| Pixel average precision | **0.5295** |
| Image F1 at validation threshold | **0.9304** |
| Pixel F1 at validation threshold | **0.3238** |

### Per-category results

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Image F1 |
|---|---:|---:|---:|---:|---:|
| bottle | 0.9992 | 0.9998 | 0.9877 | 0.7698 | 0.9921 |
| cable | 0.9799 | 0.9888 | 0.9763 | 0.6370 | 0.9392 |
| capsule | 0.9729 | 0.9939 | 0.9816 | 0.4595 | 0.9000 |
| carpet | 1.0000 | 1.0000 | 0.9921 | 0.5926 | 0.9944 |
| grid | 0.9916 | 0.9974 | 0.9886 | 0.3384 | 0.9573 |
| hazelnut | 0.9932 | 0.9961 | 0.9936 | 0.7062 | 0.9583 |
| leather | 1.0000 | 1.0000 | 0.9913 | 0.3843 | 1.0000 |
| metal_nut | 1.0000 | 1.0000 | 0.9621 | 0.6780 | 0.9841 |
| pill | 0.9853 | 0.9974 | 0.9559 | 0.6129 | 0.8201 |
| screw | 0.7998 | 0.9250 | 0.8675 | 0.2135 | 0.7113 |
| tile | 1.0000 | 1.0000 | 0.9631 | 0.5772 | 1.0000 |
| toothbrush | 0.9361 | 0.9705 | 0.9856 | 0.3897 | 0.9524 |
| transistor | 0.9229 | 0.9121 | 0.9460 | 0.6152 | 0.8333 |
| wood | 0.9982 | 0.9995 | 0.9538 | 0.6011 | 0.9524 |
| zipper | 0.9945 | 0.9986 | 0.9253 | 0.3676 | 0.9607 |

These are macro averages, so every category has equal weight. Ranking metrics and
threshold-dependent metrics answer different questions: AUROC/AP assess score
ordering across all possible thresholds, while F1 describes decisions at the
normal-only validation threshold.

## Key engineering decisions

### Multi-layer feature fusion

Intermediate transformer features retain local texture details, while final
features provide broader context. Equal normalization before concatenation keeps
one layer from dominating because of its numerical scale.

### Approximate greedy coreset

Retaining every normal patch would increase storage and nearest-neighbor cost.
Greedy k-center selection covers diverse regions of normal feature space better
than an equally sized random subset. Seeded candidate limiting and random
projection make selection tractable; checkpoints retain the original features,
not their projections. Each category checkpoint is approximately 6 MB.

### Test-independent threshold calibration

AUROC does not provide a production decision rule. The detector calibrates its
image threshold using only held-out normal training images. A finite-sample
correction targets a chosen normal false-positive rate; for small calibration
sets, the maximum observed score is used and the requested rate cannot be
formally guaranteed. The pixel threshold is explicitly treated as an empirical
quantile because spatial pixels are correlated.

### Reusable benchmark orchestration

The benchmark command discovers the official category set, reuses one loaded
encoder, creates category-specific detectors, continues while recording category
failures, and writes a machine-readable full report with macro averages.

## Reliability and leakage controls

- Training discovery is restricted to `train/good`.
- Fit and calibration indices are generated deterministically from a fixed seed.
- The memory bank sees only the 90% fitting partition.
- Thresholds see only the held-out normal 10% partition.
- Test masks and labels are accessed only during final evaluation.
- Checkpoints record model, image size, feature layers, category, sampling
  method, memory bank, and calibrated thresholds.
- Loading rejects incompatible model, resolution, or layer configurations.
- Gradient calculation is disabled throughout feature extraction and inference.
- Quantitative reports are accompanied by input/heatmap/mask visualizations.

## Interactive demonstration

The Streamlit application lets a reviewer select a fitted category, upload up to
eight images, and inspect the input, anomaly heatmap, overlay, score, calibrated
threshold, and decision. It also presents the complete benchmark table and
method limitations, and exports predictions as JSON. One frozen DINOv2 encoder
is cached and shared by every category detector; switching categories adds only
the category's memory bank and thresholds rather than another encoder copy. The
interface reports its device and inference latency. Headless Streamlit tests
verify both interface rendering and shared-encoder identity.

## Interpretation and error analysis

The macro image AUROC of 0.9716 shows strong overall ranking across diverse
objects and textures. The 0.9304 macro image F1 demonstrates that normal-only
threshold calibration also yields a useful operating point without tuning on
test defects.

Performance is not uniform. `screw` is the hardest category at 0.7998 image
AUROC, while several categories achieve perfect image ranking. Pixel AP and F1
remain lower than image metrics because defects occupy few pixels and a 16×16
token grid produces coarse boundaries. These results motivate per-category
review instead of relying only on a single benchmark average.

## Current limitations

- Coreset and multi-layer choices have not yet been isolated in a controlled
  ablation against random and single-layer baselines.
- A normal-only calibration set controls false alarms but cannot directly
  optimize the business cost of missed defects.
- The 16×16 patch grid limits exact localization of small defects.
- Latency and throughput have not been benchmarked across target hardware.
- Experiment tracking, CI, containerization, and public demo hosting are not yet
  part of the repository.
- MVTec is a clean benchmark; real production data would introduce camera,
  lighting, product-version, and process drift.

## Reproduction

Run one category:

```bash
python main.py run --category grid
```

Run the complete benchmark:

```bash
python main.py benchmark \
  --categories all \
  --experiment-dir artifacts/all_categories
```

The full run writes one detector and evaluation directory per category, plus:

- `artifacts/all_categories/summary.json`;
- `artifacts/all_categories/summary.csv`.

## Suggested CV bullets

- Built a one-class industrial anomaly-detection system using multi-layer
  DINOv2 patch embeddings, greedy k-center coreset selection, and cosine
  nearest-neighbor retrieval without anomalous training examples.
- Achieved 0.972 macro image AUROC, 0.985 macro image average precision, and
  0.965 macro pixel AUROC across all 15 MVTec AD categories.
- Designed normal-only, finite-sample-corrected threshold calibration, reaching
  0.930 macro image F1 without using test labels for model fitting or threshold
  selection.
- Engineered a reproducible PyTorch benchmark pipeline with per-category
  checkpoints, leakage controls, qualitative heatmaps, failure capture, and
  aggregate CSV/JSON reporting.

## Interview discussion points

### Why DINOv2?

DINOv2 provides strong local visual representations without requiring defect
labels. This fits manufacturing, where normal images are common but known defects
do not cover every future failure mode.

### Why not fine-tune immediately?

The category datasets are small. A frozen encoder gives a stable baseline and
reduces overfitting risk. Fine-tuning should be justified against this measured
reference through a controlled experiment.

### Why use a coreset?

Nearest-neighbor cost grows with the memory bank. The coreset reduces the bank to
2,048 diverse patches per category, yielding approximately 6 MB checkpoints
while preserving broad coverage of normal features.

### How did you avoid test leakage?

The normal training set is split before fitting. The memory bank uses 90%, the
threshold uses the held-out 10%, and test labels are read only after both have
been frozen for final metrics.

### What would you do next?

I would run coreset and layer ablations, test 448×448 inputs, add experiment
tracking and CI, benchmark latency, calibrate operating points with production
costs, and monitor score distributions for drift after deployment.
