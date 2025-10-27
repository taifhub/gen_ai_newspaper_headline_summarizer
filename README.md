# News Headline Rewriter / Summarizer

A simple but powerful project that uses fine-tuned T5-small model to generate catchy news headlines with LangChain integration for flexible prompting styles.

## Overview

This project demonstrates:
- Fine-tuning T5-small on the CNN/DailyMail dataset
- Generating news headlines from articles
- LangChain integration for flexible prompting
- Multiple headline styles (exciting, formal, short, detailed, neutral)
- Comparison between base and fine-tuned models

## Project Structure

```
newspaper_headline_summarizer/
├── README.md                      # This file
├── requirements.txt               # Python dependencies (pip)
├── environment.yml                # Conda environment file
├── train_model.ipynb              # Fine-tuning notebook
├── headline_summarizer.ipynb      # Main demo notebook
├── models/                        # Saved model checkpoints (created during training)
└── data/                          # Dataset cache (created automatically)
```

## Setup Instructions

### 1. Install Dependencies

#### Option A: Using Conda (Recommended)

Create and activate the conda environment:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate newspaper-headline-summarizer
```

#### Option B: Using pip

```bash
pip install -r requirements.txt
```

This will install:
- transformers (Hugging Face models)
- datasets (CNN/DailyMail dataset)
- torch (PyTorch)
- langchain (flexible prompting)
- jupyter (notebook environment)
- accelerate (efficient training)

### 2. Train the Model

Make sure you're in the activated conda environment, then open and run `train_model.ipynb`:

```bash
# If using conda, ensure environment is activated
conda activate newspaper-headline-summarizer

# Start Jupyter
jupyter notebook train_model.ipynb
```

The training process:
- Loads 10,000 samples from CNN/DailyMail dataset
- Fine-tunes T5-small for 3 epochs (~30-60 minutes)
- Saves the model to `./models/t5-small-headlines-final`

**Note**: Training can be done on CPU or GPU. GPU is recommended for faster training.

### 3. Generate Headlines

Open and run `headline_summarizer.ipynb`:

```bash
# Ensure environment is activated
conda activate newspaper-headline-summarizer

# Start Jupyter
jupyter notebook headline_summarizer.ipynb
```

The demo notebook includes:
- Basic headline generation
- LangChain integration
- Multiple style examples (exciting, formal, short, detailed)
- Interactive headline generator
- Comparison with base T5-small model

## Usage Examples

### Basic Headline Generation

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

model = T5ForConditionalGeneration.from_pretrained("./models/t5-small-headlines-final")
tokenizer = T5Tokenizer.from_pretrained("./models/t5-small-headlines-final")

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

article = "Your news article text here..."
result = summarizer(article, max_length=128)
print(result[0]['summary_text'])
```

### Different Styles

```python
# Exciting headline (higher temperature)
exciting_headline = styler.exciting(article)

# Formal headline (lower temperature)
formal_headline = styler.formal(article)

# Short headline (lower max_length)
short_headline = styler.short(article)
```

## Technical Details

- **Model**: T5-small (60M parameters)
- **Dataset**: CNN/DailyMail v3.0.0 (10K training samples, 1K validation)
- **Training**: 3 epochs with learning rate 3e-4
- **Framework**: PyTorch + Hugging Face Transformers
- **LangChain**: Local models only, no API keys required

## Features

✅ Fine-tuning on CNN/DailyMail dataset  
✅ Multiple headline styles  
✅ LangChain integration  
✅ Local execution (no API keys needed)  
✅ Comparison with base model  
✅ Interactive demos  
✅ Well-documented notebooks  

## Next Steps

- Experiment with different temperature values
- Fine-tune on domain-specific news (sports, tech, etc.)
- Add more sophisticated prompt engineering
- Deploy as a web service using Flask or Streamlit

## Requirements

- **Python**: 3.9+ (specified in environment.yml)
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: ~5GB disk space for models and dataset
- **GPU**: Optional but recommended for training (CUDA 11.8+)
- **Conda**: For environment management (recommended)

### Environment Management

The project includes both conda and pip setup options:

- **`environment.yml`**: Complete conda environment with PyTorch, CUDA support, and all dependencies
- **`requirements.txt`**: Python-only dependencies for pip installation

**Conda is recommended** as it handles PyTorch and CUDA dependencies more reliably.

## Troubleshooting

### Common Issues

1. **CUDA not available**: If you don't have CUDA, edit `environment.yml` and comment out the `pytorch-cuda=11.8` line
2. **Environment creation fails**: Try updating conda first: `conda update conda`
3. **Package conflicts**: Use `conda env create -f environment.yml --force` to recreate the environment
4. **Memory issues**: Reduce batch size in training arguments or use CPU-only training

### Environment Commands

```bash
# List environments
conda env list

# Remove environment (if needed)
conda env remove -n newspaper-headline-summarizer

# Update environment
conda env update -f environment.yml
```

## License

This is a simple educational project. Feel free to use and modify as needed.
