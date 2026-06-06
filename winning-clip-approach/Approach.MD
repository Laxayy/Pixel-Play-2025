## Phase 2: CLIP Approach (Score: 96)

After my ResNet-based approach plateaued at a score of 60, I started looking for methods that were better suited for zero-shot classification. The main limitation of my previous approach was that it tried to explicitly predict 85 predefined attributes before mapping them to animal classes. While this worked reasonably well, it struggled to generalize to unseen classes.

I decided to experiment with CLIP (Contrastive Language–Image Pre-training), which takes a different approach. Instead of predicting attributes, CLIP learns a shared embedding space for images and text. This allows images to be compared directly against textual descriptions of candidate classes.

The final solution used OpenAI's `clip-vit-large-patch14` model and achieved a score of **96**.

---

### 1. Loading the Model

I used the Hugging Face `transformers` implementation of CLIP.

```python
model_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
```

This handled both image preprocessing and text tokenization, allowing me to focus on the classification pipeline rather than model implementation details.

---

### 2. Prompt Ensembling

My first experiment was straightforward: I compared each image against the raw animal class names. This already achieved a score of around **90**.

To improve performance further, I generated multiple natural-language descriptions for each animal and combined their embeddings. This provided richer textual context than a single class name.

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

Using multiple prompt templates significantly improved the quality of the text representations.

---

### 3. Creating Text Embeddings

All generated prompts were passed through CLIP's text encoder to obtain embeddings. Since each animal had five prompt templates, I averaged the resulting embeddings to create a single representation per animal class.

```python
with torch.no_grad():
    text_inputs = processor(
        text=text_prompts,
        return_tensors="pt",
        padding=True
    ).to(device)

    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = normalize_embeddings(text_embeddings)

    num_templates = len(text_prompts) // len(animal_classes)

    aggregated_embeddings = (
        text_embeddings
        .view(len(animal_classes), num_templates, -1)
        .mean(dim=1)
    )

    aggregated_embeddings = normalize_embeddings(
        aggregated_embeddings
    )
```

Averaging embeddings from multiple prompts produced more stable class representations and improved classification accuracy.

---

### 4. Classification

For each test image, I generated an image embedding using CLIP's image encoder. Classification was then performed by computing similarity scores between the image embedding and the class embeddings.

```python
with torch.no_grad():
    image_inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    image_embeddings = model.get_image_features(**image_inputs)
    image_embeddings = normalize_embeddings(image_embeddings)

    similarity_scores = (
        image_embeddings @ aggregated_embeddings.T
    )

    best_match_idx = similarity_scores.argmax(dim=-1).item()
    predicted_class = animal_classes[best_match_idx]
```

The class with the highest similarity score was selected as the prediction.

---

### What I Learned

The biggest lesson from this phase was that choosing the right architecture can matter more than endlessly tuning an existing approach.

Moving from a manually designed attribute-prediction framework to a model that was already trained to align images and text resulted in a significant improvement. The final jump from **90 to 96** came entirely from prompt ensembling, showing how much impact text representation can have in zero-shot classification tasks.
