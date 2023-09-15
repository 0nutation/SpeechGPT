import os
import numpy as np
import gradio as gr
from speechgpt.utils.speech2unit.speech2unit import Speech2Unit
from speechgpt.src.infer.cli_inference import SpeechGPTInference
import soundfile as sf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model-name-or-path", type=str, default="")
parser.add_argument("--lora-weights", type=str, default=None)
parser.add_argument("--s2u-dir", type=str, default="speechgpt/utils/speech2unit/")
parser.add_argument("--vocoder-dir", type=str, default="speechgpt/utils/vocoder/")
parser.add_argument("--output-dir", type=str, default="speechgpt/output/")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

infer = SpeechGPTInference(
    args.model_name_or_path,
    args.lora_weights,
    args.load_8bit,
    args.s2u_dir,
    args.vocoder_dir,
    args.output_dir
)

def speech_dialogue(audio):
    sr, data = audio
    sf.write(
        args.input_path,
        data,
        sr,
    )
    prompts = [args.input_path]
    sr, wav = infer(prompts)
    return (sr, wav)


demo = gr.Interface(    
        fn=speech_dialogue, 
        inputs="microphone", 
        outputs="audio", 
        title="SpeechGPT",
        cache_examples=False
        )
demo.launch(share=True)



# base="s2_7B_from3750_0820"
# stage="s3_7B_from16803_COMss_0905"
# python3 speechgpt/src/infer/web_inference.py \
# --model-name-or-path "speechgpt/output/7B/v1.5/${base}/checkpoint-16803" \
# --lora-weights "speechgpt/output/7B/v1.5/${stage}/checkpoint-3300" \
# --output-dir "speechgpt/output/${stage}/" \
# --mode interactive