# Text Summarization Application

This project is a **Text Summarization Application** built using **Gradio** and **Hugging Face Transformers**. It leverages a pre-trained model, `distilbart-cnn-12-6`, fine-tuned for text summarization tasks to summarize input text effectively.

## Features

- **User-Friendly Interface**: Built with Gradio, the application provides an easy-to-use graphical interface for summarizing text.
- **High-Quality Summarization**: Utilizes the `distilbart-cnn-12-6` model from Hugging Face's Transformers library.
- **Custom Model Support**: Loads a local model for flexibility and offline usage.
- **Efficient**: Processes input quickly while maintaining high-quality results.

## Requirements

Before running the application, ensure you have the following installed:

- Python (>= 3.8)
- `torch`
- `transformers`
- `gradio`

You can install the required Python libraries using the following command:

```bash
pip install torch transformers gradio
