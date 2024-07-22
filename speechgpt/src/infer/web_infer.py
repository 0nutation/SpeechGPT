import os
import numpy as np
import gradio as gr
from speechgpt.utils.speech2unit.speech2unit import Speech2Unit
from speechgpt.src.infer.cli_infer import SpeechGPTInference
import soundfile as sf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-name-or-path", type=str, default="")
parser.add_argument("-l", "--lora-weights", type=str, default=None)
parser.add_argument("-s", "--s2u-dir", type=str, default="speechgpt/utils/speech2unit/")
parser.add_argument("-v", "--vocoder-dir", type=str, default="speechgpt/utils/vocoder/")
parser.add_argument("-o", "--output-dir", type=str, default="speechgpt/output/")
parser.add_argument("-i", "--input-dir", type=str, default="speechgpt/input/")
parser.add_argument("-h", "--https-dir", type=str, default="speechgpt/input/")
parser.add_argument("-p", "--port", type=int, default=8001)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

infer = SpeechGPTInference(
    args.model_name_or_path,
    args.lora_weights,
    args.s2u_dir,
    args.vocoder_dir,
    args.output_dir
)

def speech_dialogue(audio):
    sr, data = audio
    sf.write(
        args.input_dir,
        data,
        sr,
    )
    prompts = [args.input_dir]
    sr, wav = infer(prompts)
    return (sr, wav)


demo = gr.Interface(    
        fn=speech_dialogue, 
        inputs="microphone", 
        outputs="audio", 
        title="xtpan",
        cache_examples=False
        )

demo.launch(share=True,
            server_name="0.0.0.0",
            server_port=args.port,
            ssl_verify=False,
            ssl_certfile=args.https_dir + "cert.pem",
            ssl_keyfile=args.https_dir + "key.pem")