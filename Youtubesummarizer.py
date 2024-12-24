from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound
import re
import torch
import gradio as gr
from transformers import pipeline

model_path=r"D:\Gen_AI project\Text_summary\Text_Summarizer\models--facebook--bart-large-cnn\snapshots\37f520fa929c961707657b28798b30c003dd100b"
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)
def summary(input):
    output=text_summary(input)
    return output[0]['summary_text']



def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    pattern = r"(?:v=|be/|embed/|youtu\.be/|v/|shorts/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_transcript(url):
    """Fetches the transcript of a YouTube video."""
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL. Please check the format."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        summary_text=summary(transcript_text)
        return summary_text
    except TranscriptsDisabled:
        return "The transcript is disabled for this video."
    except VideoUnavailable:
        return "The video is unavailable."
    except NoTranscriptFound:
        return "No transcript is available for this video."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    gr.close_all()
    demo=gr.Interface(fn=get_transcript,inputs=[gr.Textbox(label="Youtube url to summarize",lines=1)],
    outputs=[gr.Textbox(label="Summarized text",lines=4)],
    title="Youtube video script summarizer",
    description="This application is used to summarize youtube video")
    demo.launch(share=True)



