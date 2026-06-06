## Phase 1: ResNet-Based Approach (Score: 60)

This was my first attempt at tackling the challenge. Since the test set contained animal classes that were never seen during training, a standard image classification pipeline was not applicable. Instead, I approached the problem as an attribute prediction task: given an image, predict its 85 animal predicates and use those predicates to identify the most likely class.

Although this approach ultimately peaked at a score of **60**, it helped me understand the dataset and taught me most of the PyTorch fundamentals that I used later in the competition.

---

### 1. Working with the Predicate Matrix

One of the first things I had to understand was the predicate data provided with the challenge. The continuous predicate matrix contained positive values representing attribute strengths, while negative values indicated the absence of a feature.

For preprocessing, I normalized only the positive values and left the negative values unchanged.

```python
self.predicate_continuous_mat = np.array(
    np.genfromtxt(predicate_file, dtype="float32")
)

# Normalize positive attribute values
self.predicate_continuous_mat[
    self.predicate_continuous_mat > 0
] /= 100.0
```

At the time, my reasoning was that negative values represented missing attributes rather than feature magnitudes, so scaling them did not seem meaningful.

---

### 2. Preserving Image Aspect Ratio

Many images in the dataset were rectangular. Directly resizing them to 224×224 distorted the animals and could potentially affect feature extraction.

To avoid this, I implemented a custom transform that padded images to a square shape before resizing.

```python
class PadToSquare:
    def __call__(self, image):
        width, height = image.size
        max_side = max(width, height)

        padding = (
            (max_side - width) // 2,
            (max_side - height) // 2,
            (max_side - width + 1) // 2,
            (max_side - height + 1) // 2,
        )

        return transforms.functional.pad(
            image,
            padding,
            fill=0,
            padding_mode="constant",
        )
```

This preserved the original aspect ratio while ensuring that all images could be processed by the network.

---

### 3. Modifying ResNet-50

I used a pre-trained ResNet-50 as the backbone model. Instead of predicting classes directly, I removed the original classification layer and added two task-specific layers.

The first layer predicted the 85 predicates, while the second mapped those predicate representations to animal classes.

```python
self.resnet = torchvision.models.resnet50(
    pretrained=True
)

self.resnet.fc = nn.Identity()

self.fc_predicates = nn.Linear(
    2048,
    num_predicates
)

self.fc_classes = nn.Linear(
    num_predicates,
    num_classes
)
```

The idea was to force the model to learn semantic attributes before making a class prediction.

---

### 4. Combining Predicate and Classification Losses

Since the network was optimizing two objectives simultaneously, I used separate loss functions for each task.

* **Label Smoothing Cross Entropy** for class prediction
* **Visual Semantic Embedding (VSE) Loss** for predicate prediction

The final training loss was a weighted combination of both.

```python
loss = (
    0.5 * loss_predicates
    + 0.5 * loss_classes
)
```

I experimented with different weightings during training, but an equal contribution from both losses gave the most stable results.

**Note:** I also used `argparse` directly inside the notebook to manage hyperparameters. While this is uncommon in Jupyter notebooks, it made it easier to keep track of experiments and avoid manually editing values between runs.

---

### 5. Attribute-Based Inference

During inference, the model produced a predicate vector rather than a direct class prediction.

To determine the final class, I compared the predicted predicate vector against the attribute vectors of all candidate classes using cosine similarity. Since all vectors were L2-normalized, this could be computed using dot products.

```python
seen_similarity = np.dot(
    seen_weights,
    predicates
)

unseen_similarity = np.dot(
    unseen_weights,
    predicates
)

if np.max(seen_similarity) > np.max(unseen_similarity):
    return np.argmax(seen_similarity)
else:
    return 40 + np.argmax(unseen_similarity)
```

The class with the highest similarity score was selected as the prediction.

---

### What I Learned

Most of my learning happened during this phase. Building this pipeline forced me to understand:

* Custom datasets and dataloaders
* Image preprocessing and augmentation
* Transfer learning with pre-trained models
* Training and validation loops
* Loss function design
* Hyperparameter tuning
* Overfitting and model evaluation

Although the final score plateaued at **60**, the experience was valuable because it exposed the limitations of manually mapping images to a fixed set of attributes. That realization eventually led me to explore CLIP and other zero-shot learning methods, which became the basis of the final winning solution.
