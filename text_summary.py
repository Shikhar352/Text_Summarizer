import torch
import gradio as gr
# Use a pipeline as a high-level helper
from transformers import pipeline
model_path=r"D:\Gen_AI project\Text_summary\models--sshleifer--distilbart-cnn-12-6\snapshots\a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)
# text = """
# Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX; early investor, CEO, and product architect of Tesla, Inc.; founder of The Boring Company; co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI. Musk is one of the richest people in the world.
# """
# print(text_summary(text))


def summary(input):
    output=text_summary(input)
    return output[0]['summary_text']
gr.close_all()

demo=gr.Interface(fn=summary,inputs=[gr.Textbox(label="input text to summarize", lines=6)],
outputs=[gr.Textbox(label="Summarized text",lines=4)],
title="Text Summarization",
description="This application is used to summarize the text")
demo.launch()