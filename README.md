# Pixel-Play-2025

# 🏆 Pixel Play'25 Challenge – 1st Place Solution

**Final Rank:** 1st out of 60 teams  
**Challenge Theme:** The Case of Sherlock Holmes and the Lost Bestiary (Zero-Shot Image Classification)

This repository contains my solution for Pixel Play'25. It was also my first machine learning project and hackathon experience.

The challenge was a Zero-Shot Learning (ZSL) task where the test images belonged to animal classes that were never seen during training. Since the model could not directly learn those classes, the goal was to use auxiliary information (animal attributes/predicates) to identify unseen animals.

The repository is divided into two notebooks that document my progression throughout the competition—from an initial transfer-learning approach that achieved a score of 60 to the final CLIP-based solution that scored 96 and secured first place.

---

## 📂 Repository Structure

- `60-pixelplay.ipynb` – Initial experiments using Transfer Learning (ResNet-50), predicate prediction, and custom loss functions.
- `96-final-pixelplay.ipynb` – Final winning solution based on OpenAI's CLIP model and prompt engineering.
- `predictions.csv` / `sorted_predictions.csv` – Final submission files.

---

## Phase 1: Learning the Fundamentals (Score: 60)

My first challenge was understanding the dataset. We were provided with:

- 50 animal classes
- 85 predicates (attributes such as "has stripes", "is aquatic", etc.)
- Binary predicate matrices
- Continuous predicate matrices

A major part of this phase was figuring out how these attributes could be used to bridge the gap between seen and unseen classes.

### What I Tried

#### Transfer Learning

I used a pre-trained ResNet-50 as the backbone model. The final classification layer was removed and replaced with:

1. A predicate prediction head for the 85 attributes.
2. A mapping layer that connected attribute predictions to animal classes.

#### Building the Pipeline

Since this was my first ML project, a large portion of the work involved learning the basics:

- Data preprocessing and normalization
- Image resizing while preserving aspect ratio
- Custom padding strategies
- Training and validation loops
- Overfitting detection and debugging

#### Loss Function Experiments

I experimented with multiple loss functions, including:

- Mean Squared Error (MSE)
- Label Smoothing Cross Entropy
- Visual Semantic Embedding (VSE) Loss
- Other custom combinations

The goal was to improve the mapping between image features and the provided attribute representations.

### Result

Despite extensive tuning, the best score I achieved with this approach was **60**. At that point, it became clear that a different strategy would be needed.

---

## Phase 2: Switching to CLIP (Score: 96)

While researching Zero-Shot Learning methods, I came across CLIP (Contrastive Language–Image Pre-training).

CLIP uses:

1. An image encoder that converts images into embeddings.
2. A text encoder that converts text descriptions into embeddings.

Both encoders project their outputs into the same feature space, allowing images and text to be compared directly.

Instead of forcing a model to predict 85 manually defined predicates, CLIP allowed me to compare an image directly against textual descriptions of animal classes.

### Prompt Engineering

Using CLIP with only animal names immediately improved my score from 60 to around 90.

The final improvement came from prompt engineering. Rather than using only class names, I generated multiple natural-language descriptions for each animal and averaged their embeddings.

```python
def generate_text_prompts(animal_classes):
    templates = [
        "A photo of a {}.",
        "A realistic photo of a {}.",
        "An image of a {} in the wild.",
        "A {} in its natural habitat.",
        "A close-up of a {}."
    ]

    text_prompts = [
        template.format(animal)
        for animal in animal_classes
        for template in templates
    ]

    return text_prompts
```

Using multiple prompt templates produced stronger text representations and increased the final score from **90 to 96**, which was enough to secure first place.

---

## Key Takeaways

This project took me from learning the basics of PyTorch and data pipelines to working with modern vision-language models.

A lesson I learned during the competition is that improving performance is not always about building a more complex model. Sometimes the biggest gains come from understanding the problem properly and choosing an architecture that is naturally suited to it.

---

## Final Score

🏆 **96/100 – 1st Place**
