# Learning Guide: DINOv2 for Visual Anomaly Detection

This guide explains the project from first principles. It assumes that you know
basic Python but does not assume prior knowledge of anomaly detection,
transformers, or industrial machine learning.

## 1. What problem are we solving?

Visual anomaly detection asks two related questions:

1. **Detection:** Does this image contain a defect?
2. **Localization:** Which pixels contain the defect?

For example, a factory may produce thousands of visually similar metal parts.
Most parts are normal, while defects such as scratches or contamination are rare.
A useful system should assign each image an anomaly score and produce a heatmap
showing suspicious regions.

This differs from ordinary image classification. A classifier normally learns
from examples of every class, such as many images of both good and scratched
parts. In industrial anomaly detection, we often have many normal samples but
few—or no—examples of future defects. The model must therefore learn what
“normal” looks like and identify deviations from it.

This is called **one-class** or **unsupervised anomaly detection**. “Unsupervised”
here means defect labels are not used to fit the detector. The pretrained DINOv2
encoder was originally trained on a large external image collection using
self-supervision.

## 2. The MVTec AD dataset

MVTec AD is a benchmark for industrial anomaly detection. It contains object and
texture categories. Each category follows this structure:

```text
category/
├── train/
│   └── good/                 # only normal training images
├── test/
│   ├── good/                 # normal test images
│   └── <defect_type>/        # anomalous test images
└── ground_truth/
    └── <defect_type>/        # binary pixel masks
```

This project evaluates all 15 MVTec AD categories. The `grid` category, for
example, contains:

- 264 normal training images
- 78 test images
- five defect types: bent, broken, glue, metal contamination, and thread

The test masks contain `1` for defective pixels and `0` elsewhere. They are used
only to evaluate localization, never to build the memory bank.

### Why keeping test data separate matters

Using test labels to choose model settings gives an unrealistically optimistic
result. This is called **data leakage**. In a production experiment, the dataset
would normally be divided into:

- training data for fitting;
- validation data for model selection and threshold calibration;
- test data used once for final reporting.

MVTec provides no official validation split. This project deterministically
reserves 10% of each category's normal training images for calibration. No test
image or test label participates in fitting or threshold selection.

## 3. Images as numbers

A color image is a tensor with three channels: red, green, and blue. A batch of
images typically has the PyTorch shape:

```text
[batch size, channels, height, width]
```

For eight 224×224 RGB images, this is `[8, 3, 224, 224]`.

Here, a **batch** is simply a group of images processed by the model at the same
time. The four dimensions mean:

| Dimension | Meaning |
|---|---|
| `8` | eight separate images |
| `3` | red, green, and blue channels |
| `224` | image height |
| `224` | image width |

Batching lets PyTorch perform the same matrix operations for several images in
parallel. It does not mix the images together; every output remains associated
with its original input. If only three images remain at the end of a dataset,
the final input batch has shape `[3, 3, 224, 224]`.

Before entering DINOv2, each image is:

1. resized to a fixed resolution;
2. converted to floating-point values;
3. scaled from integer pixel values into approximately `[0, 1]`;
4. normalized using ImageNet channel means and standard deviations.

Normalization places inputs in the numerical range expected by the pretrained
encoder. Using different preprocessing at training and inference is a common and
serious ML bug.

## 4. From pixels to features

Raw pixels are not always a useful way to compare images. A small lighting or
position change can alter many pixels even though the manufactured part remains
normal. A neural network converts pixels into **features**: numerical vectors
that represent patterns such as texture, shape, edges, and semantic content.

A feature vector is also called an **embedding**. Similar visual content should
have embeddings that are close together, while different content should be
farther apart.

An embedding is simply a list of numbers. A tiny illustrative embedding might
look like:

```text
[0.12, -0.04, 0.31, ..., 0.08]
```

An individual number does not have a simple interpretation such as “scratch” or
“metal.” Together, the values encode visual properties such as edges, texture,
shape, color patterns, object structure, and surrounding context. This lets the
detector compare visual meaning rather than comparing raw pixels directly.

This project uses DINOv2 as a feature extractor. Its parameters are frozen, so
we do not change them during fitting.

## 5. What is a Vision Transformer?

DINOv2 uses a Vision Transformer, or ViT. A ViT processes an image in roughly
the following way:

1. Split the image into fixed-size square patches.
2. Convert every patch into a vector called a patch token.
3. Add information describing each patch's position.
4. Pass all tokens through transformer blocks.
5. Use self-attention to let patches exchange information.

The default model in this project is ViT-S/14:

- `S` means the small model variant.
- `14` means each input patch is 14×14 pixels.

At an input size of 224×224, the patch grid is:

```text
224 / 14 = 16 patches per side
16 × 16 = 256 patch tokens per image
```

Each layer's token has 384 feature values in ViT-S/14. This project extracts
blocks 8 and 11, normalizes them independently, concatenates them, and normalizes
again. The resulting descriptor has shape `[batch, 256, 768]`. Intermediate
features retain more local texture detail, while later features contain more
global and semantic context.

For a batch of eight images, the individual and combined shapes are:

```text
Block 8:          [8 images, 256 patches, 384 features]
Block 11:         [8 images, 256 patches, 384 features]
Combined output:  [8 images, 256 patches, 768 features]
```

The model therefore produces `8 × 256 = 2,048` patch embeddings for this batch,
while still preserving which image and spatial position each embedding belongs
to.

### Why blocks 8 and 11?

ViT-S/14 contains 12 transformer blocks, numbered 0 through 11. Representations
generally progress from local visual patterns toward increasingly global and
semantic context:

```text
early blocks        middle/later blocks       final block
edges and texture → shapes and structure → global context
```

Block 11 supplies the final context-rich representation. Block 8 is a relatively
late intermediate layer that can retain more local texture and spatial detail
than the final block while still representing meaningful object structure. The
current design concatenates them to give anomaly detection access to both kinds
of information.

Block 8 is an engineering hypothesis, not a proven optimum. A controlled
ablation should compare block 11 alone, blocks 6+11, blocks 8+11, and possibly
several other combinations using the same split, seed, coreset size, and image
resolution. Image and pixel metrics, latency, and checkpoint size should all be
reported. Until that experiment is complete, the precise claim is that block 8
is a reasonable complementary intermediate layer—not that it is universally the
best layer.

### Why normalize the embeddings?

L2 normalization changes a vector's length to one while preserving its
direction. For a two-value example:

```text
vector = [3, 4]
length = √(3² + 4²) = 5
normalized vector = [3/5, 4/5] = [0.6, 0.8]
```

Without normalization, a vector with a larger numerical magnitude could appear
more important even when its visual meaning is not more relevant. After
normalization, cosine similarity can be computed as a dot product and focuses on
the directions of the descriptors. This project normalizes each layer before
concatenation so one layer cannot dominate solely through scale, then normalizes
the combined 768-dimensional descriptor again.

### Self-attention intuition

Self-attention allows one patch to use information from other patches. A patch
covering part of the grid texture can compare itself with surrounding patches
and learn whether it fits the larger pattern. This global context is one reason
transformer features work well for visual recognition.

## 6. What is DINOv2?

DINOv2 is a self-supervised vision model. **Self-supervised learning** constructs
a learning signal from the data itself instead of relying on manually assigned
class labels.

At a high level, DINO-style training presents multiple transformed views of an
image to teacher and student networks. The student learns to produce a
representation consistent with the teacher across those views. Additional
objectives encourage useful patch-level representations. The expensive
pretraining has already been performed; this project loads the resulting
weights.

This gives us **transfer learning**: knowledge learned from a large, general
dataset is reused for a smaller industrial dataset.

### DINO versus DINOv2

DINO is the earlier method. DINOv2 improves the training recipe, data curation,
and learned representations. The repository uses DINOv2 because the initial code
and cached weights target `dinov2_vits14`.

## 7. How neural networks are normally trained

A neural network contains numerical **parameters**, usually called weights. At
the beginning of ordinary supervised training, many weights are random. Training
repeatedly performs these steps:

1. **Forward pass:** send a batch of inputs through the model to make predictions.
2. **Loss calculation:** measure how different the predictions are from the
   desired answers.
3. **Backpropagation:** calculate how each weight contributed to the loss.
4. **Optimization:** slightly change the weights in a direction expected to lower
   the loss.

An **optimizer**, such as AdamW or stochastic gradient descent, controls those
updates. The **learning rate** controls their size. A **batch** is the subset of
training examples processed in one update, and an **epoch** is one pass through
the complete training dataset.

During training, the goal is not to memorize the samples but to learn patterns
that **generalize** to unseen data. A model that performs well on its training
set but poorly on new data is **overfitting**. Validation data helps select model
settings while monitoring this problem.

**Pretraining** performs this expensive learning on a large, general dataset.
**Fine-tuning** starts from pretrained weights and updates some or all of them on
a smaller target dataset. **Feature extraction** keeps the pretrained weights
fixed and uses their outputs in another method. This project currently uses
feature extraction.

PyTorch has two model modes that are easy to confuse with learning itself:
`model.train()` enables training-time behavior in certain layers, while
`model.eval()` enables inference behavior. Weight updates occur only when
gradients are calculated and an optimizer applies them. This project uses
evaluation mode and inference mode because DINOv2 remains frozen.

The difference between conventional training and this project can be summarized
as:

```text
Conventional training:
image → model → prediction → loss → backpropagation → updated weights

This project:
image → frozen DINOv2 → reusable patch embeddings
```

Freezing the encoder makes fitting faster, reduces GPU memory use, limits the
risk of overfitting small category datasets, and reuses the visual knowledge
DINOv2 acquired during large-scale pretraining. The anomaly detector still
learns category-specific state, but that state is a memory bank rather than a
new set of neural-network weights.

## 8. What does “training” mean in this project?

No gradient-based neural-network training currently occurs. The DINOv2 encoder
is frozen. Fitting the detector means building a **memory bank** of embeddings
from normal images.

For every normal fitting image:

1. extract its block-8 and block-11 patch embeddings;
2. normalize and concatenate the two layers;
3. combine embeddings from the 90% fitting partition;
4. retain 2,048 representative embeddings with coreset selection;
5. calibrate thresholds on the held-out 10%;
6. save the memory bank, configuration, and thresholds in a checkpoint.

Without sampling, this category would produce:

```text
238 fitting images × 256 patches = 60,928 candidate patch embeddings
```

Every candidate is a 768-value description of one normal region. Candidates can
represent common background, an ordinary edge, a texture intersection, normal
lighting variation, or another valid part of the product. Together they describe
the range of appearances that the detector should accept as normal.

### What is the memory bank?

The memory bank is a compact catalogue of representative normal patches:

```text
memory bank = [
    normal patch embedding 1,
    normal patch embedding 2,
    normal patch embedding 3,
    ...
]
```

At inference time, each new patch effectively asks: “Is there anything similar
to me in the catalogue of normal patches?” A close match suggests normality; no
close match suggests an anomaly. Keeping all 60,928 candidates would increase
checkpoint size and require every test patch to perform many more comparisons,
so the default bank retains 2,048 representatives.

### Why not select the bank randomly?

Random sampling can select many nearly identical patches. If a plain background
occupies most training images, a random bank may waste much of its capacity on
that background while missing uncommon but valid edges or object structures.
A coreset instead attempts to cover diverse regions of normal feature space.

The geographical analogy is choosing emergency-service locations: placing all
stations in one neighborhood is wasteful, even if that neighborhood contains
most residents. A useful selection spreads stations so that every region is
reasonably close to one.

Limiting the bank reduces checkpoint size and inference cost. The implementation
uses approximate greedy **k-center coreset selection**. It first limits the
candidate pool to 25,000 seeded samples and projects the 768-dimensional vectors
to 32 dimensions. It then repeatedly selects the candidate farthest from its
nearest selected center. The projected vectors make selection efficient, while
the checkpoint retains the corresponding full 768-dimensional embeddings.

Compared with random sampling, the coreset deliberately covers diverse regions
of the normal feature space. The approximation is a practical engineering
trade-off: exact greedy selection over every high-dimensional patch would be
much more expensive.

The greedy farthest-first procedure is:

1. Select an initial representative near the center of the projected candidates.
2. Measure every candidate's distance from its nearest selected representative.
3. Find the candidate whose nearest representative is still farthest away.
4. Add that candidate to the coreset.
5. Update the nearest distances and repeat until 2,048 points are selected.

Before this procedure, the implementation reduces the work in two ways:

```text
60,928 full 768-D embeddings
          ↓ seeded candidate limiting
25,000 full 768-D embeddings
          ↓ seeded random projection
25,000 compact 32-D embeddings
          ↓ greedy k-center selection
2,048 selected indices
          ↓ retrieve original descriptors
2,048 full 768-D memory embeddings
```

The seed makes candidate limiting and projection reproducible. Projection is
used only to make coreset selection cheaper; it does not replace the selected
features. The checkpoint stores the original 768-dimensional embeddings at the
chosen indices.

### Coreset selection from first principles

A **coreset** is a small selection intended to represent a larger collection.
Imagine choosing reference paint samples: many nearly identical whites add less
coverage than a selection that also includes grays, blues, and greens. Here,
each sample is a normal patch embedding, a numerical description of appearance.

For example, 200 fitting images produce `200 × 256 = 51,200` embeddings. Many
will describe similar backgrounds or surfaces. A hypothetical dataset might
contain 70% background patches, 25% smooth product surfaces, and 5% fine edges.
Random selection tends to follow those proportions. Coreset selection instead
tries to cover different appearances, including uncommon normal edges that
might otherwise look unfamiliar at inference time. It does not use named
appearance classes or defect labels to make these selections.

#### What does coverage mean?

Treat every embedding as a point in feature space. For each candidate, measure
its distance to the **nearest selected representative**:

```text
coverage distance = minimum distance to any selected representative
```

A small distance means that something similar is already represented. A large
distance means the candidate is poorly represented. Greedy k-center selection
repeatedly chooses the candidate with the largest coverage distance. Here, `k`
is the requested number of representatives, and a center is a selected patch
embedding. “Greedy” means choosing the next point using the current distances,
without searching all possible final subsets.

The goal is to reduce the worst distance between a candidate and its nearest
representative. The implementation approximates this goal; it does not promise
the globally optimal subset or a particular detection improvement.

#### The exact steps in `_coreset_sample()`

1. **Limit candidates when needed.** If there are more than 25,000 embeddings,
   take a seeded random subset of 25,000. An appearance omitted here cannot be
   selected later. Smaller candidate pools are kept intact.
2. **Project for cheaper selection.** Multiply the 768-dimensional vectors by
   a seeded random matrix to obtain 32-dimensional vectors, then L2-normalize
   them. This approximately preserves useful relationships while reducing the
   cost of repeated comparisons. These are the default dimensions.
3. **Choose the first representative.** Average the projected vectors, then
   choose the candidate with the highest dot product with that average. This
   starts from a relatively central appearance.
4. **Initialize nearest distances.** For every candidate, compute
   `1 - dot(candidate, first representative)` in the normalized projected space.
5. **Select the least represented candidate.** Choose the candidate with the
   largest current nearest distance.
6. **Update coverage and repeat.** Compare each candidate with the new
   representative and retain the smaller distance:

```text
updated nearest distance = min(previous nearest distance,
                               distance to new representative)
```

The minimum matters because a candidate is represented well if it is close to
**any** selected representative. Consider this illustrative update:

| Candidate | Previous nearest distance | Distance to new representative | Updated nearest distance |
|---|---:|---:|---:|
| A | 0.10 | 0.70 | 0.10 |
| B | 0.80 | 0.05 | 0.05 |
| C | 0.60 | 0.50 | 0.50 |

Among these candidates, C would be selected next: it remains farthest from its
nearest representative. B no longer needs priority because the new selection
represents it well.

Selection stops at the requested count, capped by the available candidate
count. If every candidate is needed, the implementation returns them directly.
The resulting memory bank contains original 768-dimensional normal embeddings,
not the projected vectors and not newly calculated cluster averages. At the
default size, `2,048 × 768 × 4` bytes is approximately 6 MiB of float32 feature
data.

Coreset selection happens during fitting. During inference, the detector uses
the saved bank directly; it does not rebuild the coreset for each new image.

### What is stored in a checkpoint?

Each category checkpoint contains:

- the 2,048 full memory embeddings;
- the DINOv2 model name;
- the selected transformer blocks;
- the input resolution;
- the MVTec category;
- the sampling method;
- the calibrated image threshold;
- the calibrated pixel threshold.

DINOv2's pretrained weights are loaded separately rather than duplicated inside
every category checkpoint.

### Why can every category share one encoder in the demo?

The encoder and the category detector have different jobs:

```text
Shared DINOv2 encoder
image → general-purpose patch embeddings

Category-specific state
patch embeddings → compare with that category's memory bank → decision
```

The DINOv2 weights never change when fitting `bottle`, `grid`, or `transistor`.
Consequently, loading one encoder per category would create identical copies of
the largest component. The Streamlit app instead caches one frozen encoder and
passes the same PyTorch object to every category detector.

Each category detector still has independent state: its approximately 6 MB
memory bank, image threshold, pixel threshold, and category metadata. Changing
the sidebar selection changes this lightweight state, not the neural network.
In simplified form, memory changes from roughly:

```text
Before: 15 × encoder + 15 × category state
After:   1 × encoder  + 15 × category state
```

Sharing does not change extracted features or predictions because the encoder
is frozen, evaluated in inference mode, and identical for every category. It is
a serving optimization rather than a new training method. The interface reports
the device, inference time, time per image, and detector-access time so runtime
behavior is visible rather than hidden.

This approach is non-parametric: its learned state is the stored data
representation rather than updated neural-network weights.

More precisely, a parametric learner stores what it learned in a fixed collection
of adjustable parameters. This detector stores representative data-derived
embeddings, so changing the normal dataset changes the stored reference set
itself. The frozen DINOv2 encoder is a parametric neural network, but the
category-specific anomaly detector built on top of it is non-parametric.

## 9. How inference works

For a test image, we extract normalized patch embeddings in the same way as for
the training images. Each test patch is compared with every memory-bank patch.

For normalized vectors `q` and `m`, cosine similarity is their dot product:

```text
similarity(q, m) = q · m
```

The nearest normal patch is the memory item with maximum similarity. The anomaly
score is:

```text
patch anomaly score = 1 - maximum cosine similarity
```

A patch very similar to something observed during fitting receives a score near
zero. An unfamiliar patch receives a larger score.

For example:

```text
Familiar patch:
highest normal similarity = 0.97
anomaly score = 1 - 0.97 = 0.03

Unfamiliar patch:
highest normal similarity = 0.55
anomaly score = 1 - 0.55 = 0.45
```

The second patch is more suspicious because none of the stored normal patches is
very similar to it.

The implementation processes queries in chunks. Chunking controls peak memory
use without changing the result.

## 10. From patch scores to a heatmap

### One set of patch scores, two output paths

After nearest-normal comparison, each image has 256 patch scores. A high score
means the patch differs from its best matching normal representative. That
match can come from anywhere in the category memory bank; it need not have the
same spatial position as the new patch.

The same scores answer two different questions:

| Output | Question | Calculation | Preserves location? |
|---|---|---|---|
| Image score | Does the image contain an unusual region? | Average the highest-scoring patches | No |
| Anomaly map | Where are the unusual regions? | Restore the 16×16 grid and enlarge to 224×224 | Yes |

```text
256 patch anomaly scores
          │
          ├── highest scores → average → one image score → threshold → label
          │
          └── all scores in spatial order → 16×16 grid → 224×224 anomaly map
```

The image score is computed directly from the original patch scores, before
interpolation. It is not the average of the enlarged or colored heatmap. An
anomaly score is a distance measure, not a probability: a score of `0.30` does
not mean a 30% chance of a defect.


At 224×224 resolution, DINOv2 returns a 16×16 grid of patch embeddings. DINOv2
does not produce anomaly scores itself. The detector compares every embedding
with the memory bank and converts the resulting similarities into a 16×16 grid
of patch anomaly scores. This is a coarse anomaly map. Bilinear interpolation
enlarges it to 224×224 so it can be overlaid on the input image.

A simplified score grid might contain a local high-score region:

```text
0.03  0.04  0.05  0.04
0.04  0.06  0.31  0.27
0.03  0.05  0.42  0.35
0.02  0.04  0.08  0.07
```

The larger values near the center indicate the suspicious location. The real
grid contains 16 rows and 16 columns rather than four.

Interpolation makes the visualization smoother, but it does not create new
spatial detail. Small defects may be difficult to localize because a token
represents a 14×14 region and also includes contextual information.

For intuition, if neighboring patch scores are `0.1` and `0.5`, interpolation
may draw intermediate values such as `0.2`, `0.3`, and `0.4` between them. Those
values make the heatmap smooth, but they are estimates between existing scores,
not newly detected fine-grained evidence.

### How the code preserves location

Patch features retain their grid order. The detector reshapes the scores and
then enlarges the grid, using the following logic at the default resolution:

```python
# One score at each of 256 patch positions per image.
anomaly_maps = patch_scores.reshape(batch_size, 1, 16, 16)
anomaly_maps = F.interpolate(
    anomaly_maps,
    size=(224, 224),
    mode="bilinear",
    align_corners=False,
)
```

The `1` is a single score channel, rather than the three RGB channels in an
input image. Bilinear interpolation blends nearby grid values to fill the
larger map. It adds display pixels, not independently detected evidence.
The inference display then maps values to colors and blends the colored map
with the resized input. Image classification still uses its separate image
score and threshold; the heatmap shows the spatial distribution of scores.

Increasing the input size to 448 gives a 32×32 token grid:

| Input resolution | Patch grid | Patch embeddings per image |
|---|---:|---:|
| 224×224 | 16×16 | 256 |
| 448×448 | 32×32 | 1,024 |

The higher resolution provides four times as many spatial locations and can help
localize smaller defects. It also requires more transformer computation and
memory, four times as many nearest-neighbor patch queries, and larger heatmaps.
The result remains patch-based; it is simply less coarse.

## 11. Producing one score per image

Detection needs one value for the complete image. Averaging all patches can
hide a small defect among many normal regions. Consider an image with:

```text
254 normal-looking patches: 0.02 each
  2 unusual patches:        0.70 and 0.80

Mean of all patches = (254 × 0.02 + 0.70 + 0.80) / 256
                    ≈ 0.026
```

The low overall average hides the two high scores. The project instead averages
approximately the highest-scoring 1% of patches. The exact implementation is:

```python
k = max(1, num_patches // 100)
image_scores = patch_scores.topk(k, dim=1).values.mean(dim=1)
```

Integer division rounds down, and `max(1, ...)` ensures at least one patch is
used. With 256 patches, `k = 2`, so the example becomes:

```text
Image score = (0.80 + 0.70) / 2 = 0.75
```

`dim=1` selects and averages patches within each image independently. It does
not mix scores from different images in a batch. For a batch of eight images,
this produces eight image scores.

Taking just the maximum would give `0.80`. Averaging the top two reduces
reliance on one extreme value while keeping small unusual regions influential.
It does not eliminate sensitivity to noisy patches. The aggregation is a design
choice that should be validated rather than assumed optimal.

The same example's heatmap would retain the positions of the `0.80` and `0.70`
patches. If both occurred toward the lower right, the heatmap would highlight
that region, while the image score would summarize its severity as `0.75`.
The two outputs describe the same evidence at different levels of detail.

### Selecting thresholds without test labels

A continuous anomaly score is not yet a yes/no decision. A **decision threshold**
is the boundary that converts a score into a label:

```text
score ≤ threshold → NORMAL
score > threshold → ANOMALY
```

For example, the fitted transistor checkpoint has an image threshold of about
`0.2266`. An image score of `0.18` is therefore classified as normal, while a
score of `0.30` is classified as anomalous.

The project stores two category-specific thresholds:

1. The **image threshold** decides whether the complete image is normal or
   anomalous.
2. The **pixel threshold** decides which anomaly-map locations are unusually far
   from normal patches. It can turn a continuous heatmap into a binary defect
   mask, although the demo displays continuous intensity to preserve more
   information.

Each category reserves 10% of its normal training images as a calibration set.
After fitting the memory bank on the other 90%, the detector scores these unseen
normal images and their pixels. This answers two practical questions:

```text
How high can an unseen normal image's score become?
How high can an unseen normal region's score become?
```

The image threshold uses the first distribution; the pixel threshold uses the
second. Test images and test labels do not participate in either choice.

The image threshold uses a finite-sample-corrected, conformal-style upper order
statistic. For `n` normal calibration scores and target false-positive rate `α`,
the requested rank is:

```text
rank = ceil((n + 1) × (1 - α))
```

The default `α = 0.05` targets a 5% normal-image false-positive rate. If the
requested rank is `n + 1`, the implementation uses the largest observed score
instead of an infinite threshold. This is more conservative than an ordinary
empirical quantile, but a small validation set cannot statistically guarantee
such a low false-positive rate: its finest observable tail probability is about
`1 / (n + 1)`. Pixel scores are spatially correlated, so their threshold is an
empirical 99th percentile of normal calibration pixels (`α = 0.01`), not an
independent-pixel guarantee.

Threshold-dependent precision, recall, specificity, accuracy, F1, and confusion
counts are calculated only after both thresholds have been fixed.

Normal-only calibration primarily controls false alarms. It cannot directly
choose the best trade-off between false alarms and missed defects because it has
not seen representative validation defects. A production system should refine
this statistical starting point using a separate validation set and explicit
business costs, while keeping the final test set untouched.

### Complete detector flow

```text
NORMAL TRAINING IMAGES
          │
          ├── 90% fitting images
          │       │
          │       ▼
          │   Frozen DINOv2
          │       │
          │       ▼
          │   Block 8 + block 11 embeddings
          │       │
          │       ▼
          │   Candidate normal patch embeddings
          │       │
          │       ▼
          │   Candidate limiting + random projection
          │       │
          │       ▼
          │   2,048-patch k-center coreset
          │
          └── 10% held-out normal images
                  │
                  ▼
             Normal anomaly scores
                  │
                  ▼
             Calibrated thresholds

NEW IMAGE
    │
    ▼
Frozen DINOv2 patch embeddings
    │
    ▼
Compare each patch with the normal memory bank
    │
    ├── top patch scores → image score → NORMAL or ANOMALY
    │
    └── 16×16 score grid → interpolation → 224×224 heatmap
```

## 12. Understanding the metrics

### Confusion-matrix terms

After selecting a decision threshold:

- **True positive:** a defect correctly identified as anomalous.
- **False positive:** a normal sample incorrectly flagged.
- **True negative:** a normal sample correctly accepted.
- **False negative:** a defect incorrectly accepted as normal.

False negatives may be especially costly in quality inspection, while too many
false positives slow production through unnecessary manual inspection.

### AUROC

The receiver operating characteristic curve measures the true-positive rate
against the false-positive rate across all possible thresholds. AUROC is the
area under this curve.

- `1.0` means perfect ranking.
- `0.5` is approximately random ranking.

AUROC evaluates ranking and does not supply an operating threshold.

### Average precision

Precision is the fraction of predicted anomalies that are truly anomalous.
Recall is the fraction of real anomalies found. Average precision summarizes the
precision-recall curve across thresholds.

Average precision is particularly informative when positive examples are rare.
Pixel anomalies occupy a small fraction of all pixels, which explains why pixel
average precision can be much lower than pixel AUROC.

### Image-level versus pixel-level metrics

- Image metrics compare test image scores with normal/anomalous labels.
- Pixel metrics compare every heatmap value with the corresponding binary mask.

A model may detect that an image is abnormal while imprecisely outlining the
defect, leading to strong image metrics but weaker pixel average precision.

## 13. Current measured result

The complete pipeline was executed on all 15 MVTec AD categories at 224×224:

| Metric | Value |
|---|---:|
| Macro image AUROC | 0.9716 |
| Macro image average precision | 0.9853 |
| Macro pixel AUROC | 0.9647 |
| Macro pixel average precision | 0.5295 |
| Macro image F1 at validation threshold | 0.9304 |
| Macro pixel F1 at validation threshold | 0.3238 |

Macro averaging gives every category equal weight. There were no failed
categories. `screw` was the hardest detection category (0.7998 image AUROC),
which is valuable evidence that aggregate metrics should always be accompanied
by per-category results. The complete table is stored in
`artifacts/all_categories/summary.csv` and `summary.json`.

## 14. Walking through the code

### `dino_anomaly/data.py`

`MVTecDataset` discovers images, loads RGB inputs, loads grayscale ground-truth
masks, and assigns image labels. Training is explicitly restricted to the
`train/good` directory to prevent accidental leakage.

### `dino_anomaly/model.py`

`DinoAnomalyDetector`:

- loads and freezes DINOv2, or accepts a shared frozen encoder for serving;
- extracts and fuses normalized tokens from multiple layers;
- builds a greedy k-center coreset;
- calibrates image and pixel thresholds on held-out normal data;
- computes nearest-neighbor patch distances;
- creates image scores and upsampled anomaly maps;
- saves and loads detector checkpoints.

The `@torch.inference_mode()` decorator disables gradient tracking. This reduces
memory use and makes it clear that the encoder is not being optimized.

### `dino_anomaly/evaluation.py`

The evaluation loop collects predictions, calculates ranking and
threshold-dependent metrics, writes them to JSON, and creates
input/heatmap/ground-truth comparison figures.

### `dino_anomaly/cli.py`

The CLI connects the components and supports four commands:

```bash
# Build and save the memory bank
python main.py fit --category grid

# Load a fitted checkpoint and evaluate it
python main.py evaluate --category grid

# Perform both operations
python main.py run --category grid

# Fit and evaluate all 15 categories, then aggregate their results
python main.py benchmark --categories all
```

It also selects CUDA, Apple MPS, or CPU automatically and sets random seeds.

## 15. How this resembles an industry ML workflow

An industry system is more than a model definition. This project already
separates several lifecycle stages:

```text
data loading → preprocessing → fitting → versioned artifact
                                      ↓
test images → identical preprocessing → inference → metrics and visual review
```

Important engineering ideas demonstrated here include:

- a reproducible CLI instead of notebook-only code;
- separation of data, model, evaluation, and orchestration;
- serialized model state;
- quantitative and qualitative evaluation;
- support for multiple compute devices;
- explicit prevention of training/test leakage.
- validation-calibrated operating thresholds;
- per-category artifacts and aggregate CSV/JSON benchmark reports.

Production systems would additionally track dataset and code versions, validate
inputs, monitor latency and drift, enforce automated tests, and deploy through a
repeatable build pipeline.

## 16. Limitations and next experiments

### Limited ablation evidence

The coreset and two-layer representation are implemented, but a controlled
experiment should compare them with random sampling and single-layer features
under identical seeds. Without an ablation, we cannot attribute improvements to
one component confidently.

### Low-resolution anomaly maps

Evaluate 224 and 448 input sizes. Report both localization quality and runtime so
the trade-off is visible.

### Normal-only validation

Normal-only calibration controls false alarms but cannot optimize the trade-off
against missed defects. Production threshold selection should combine this
statistical baseline with reviewed defects and explicit business costs.

### No experiment tracking yet

Record configuration, code revision, metrics, runtime, hardware, and generated
artifacts using MLflow or Weights & Biases. This prevents “mystery results” that
cannot later be reproduced.

## 17. Suggested learning roadmap

1. Run `fit`, `evaluate`, and `run` separately and inspect their outputs.
2. Print tensor shapes at each stage until the data flow feels intuitive.
3. Change the memory-bank size and measure accuracy and speed.
4. Compare 224 and 448 input resolutions.
5. Compare random and coreset sampling with a controlled ablation.
6. Compare single-layer and multi-layer descriptors.
7. Study threshold sensitivity and calibration-set size.
8. Add experiment tracking and automated tests.
9. Package inference behind an API and monitor latency and drift.

For each experiment, change one factor at a time, state a hypothesis before
running it, and record the result whether it improves or worsens. That habit is
more representative of professional ML work than simply chasing the highest
metric.

## 18. Questions you should be able to answer

After studying the project, practice explaining:

1. Why can a model detect defects without seeing defective training samples?
2. Why are patch tokens more useful than one global image token for localization?
3. Why must embeddings be normalized before cosine-distance scoring?
4. What accuracy/speed trade-off does the memory-bank size create?
5. Why can pixel AUROC be high while pixel average precision is much lower?
6. Why does a benchmark metric not automatically define a production threshold?
7. When might fine-tuning DINOv2 hurt rather than help?
8. How would you detect data drift after deployment?

If you can answer these clearly and connect them to code and measurements, you
understand the project rather than merely having run it.
