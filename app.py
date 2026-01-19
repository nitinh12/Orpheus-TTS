import os
# 👈 vLLM Length Control via Environment Variables
os.environ['VLLM_MAX_MODEL_LEN'] = '8192'
os.environ['VLLM_MAX_NUM_SEQS'] = '1'
os.environ['VLLM_MAX_NUM_BATCHED_TOKENS'] = '4096'

import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import time
import io
import numpy as np
import gc
import torch
import atexit

model = None

def cleanup_model():
    """Force unload model and clear GPU memory"""
    global model
    if model is not None:
        print("🧹 Unloading model...")
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("✅ Model unloaded and GPU cleared")

def generate_speech(prompt, voice, temperature, repetition_penalty, top_p):
    global model
    
    cleanup_model()
    
    try:
        print("🚀 Loading fresh model...")
        start_load = time.monotonic()
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        load_time = time.monotonic() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s (Max context: 8192 tokens)")
        
        gen_start = time.monotonic()
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice=voice,
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p)
        )
        
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            chunk_count = 0
            for audio_chunk in syn_tokens:
                chunk_count += 1
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
        
        gen_time = time.monotonic() - gen_start
        
        cleanup_model()
        
        audio_buffer.seek(0)
        audio_data = np.frombuffer(audio_buffer.read()[44:], dtype=np.int16)
        
        info_text = f"""✅ Generated {duration:.2f}s audio
⏱️ Load: {load_time:.2f}s | Gen: {gen_time:.2f}s
📊 RTF: {gen_time/duration:.2f}x | {chunk_count} chunks
🔧 vLLM Max Context: 8192 tokens"""
        
        return (24000, audio_data), info_text
        
    except Exception as e:
        cleanup_model()
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

atexit.register(cleanup_model)

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
EMOTIVE_TAGS = ["`<laugh>`", "`<chuckle>`", "`<sigh>`", "`<cough>`", "`<sniffle>`", "`<groan>`", "`<yawn>`", "`<gasp>`"]

with gr.Blocks(title="Orpheus TTS - vLLM Backend") as demo:
    gr.Markdown(f"""
    # 🎙️ Orpheus TTS - vLLM Backend (8192 Token Context)
    **Fresh model loaded each generation** - No crashes!
    
    **Tips**: Use {', '.join(EMOTIVE_TAGS)} or `uhm` for human-like speech
    **Length**: Longer prompts + higher temperature = longer audio
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Longer text = longer audio... Try 200+ words!",
                lines=6,
                value="Hi guys, welcome back to another video — I'm Nero from CogniCore-AI, and today we will be testing the newly released Z Image Turbo Fun ControlNet Union 2.1 models, and I will be sharing a super-fast download script to download the models on RunPod. This is perfect for anyone running AI workloads on cloud GPUs, and I'll show you exactly how to get these massive models deployed in minutes."
            )
            
            voice_dropdown = gr.Dropdown(
                choices=VOICES,
                value="leo",
                label="Voice"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature_slider = gr.Slider(0.1, 2.0, 1.0, 0.1, label="Temperature", info="Higher = faster speech")
                repetition_penalty_slider = gr.Slider(1.0, 2.0, 1.1, 0.05, label="Repetition Penalty", info=">=1.1 required")
                top_p_slider = gr.Slider(0.1, 1.0, 0.95, 0.05, label="Top P")
            
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech (Fresh Model)", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            info_output = gr.Textbox(label="Status", lines=5)
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[prompt_input, voice_dropdown, temperature_slider, repetition_penalty_slider, top_p_slider],
        outputs=[audio_output, info_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", None, ""),
        outputs=[prompt_input, audio_output, info_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True)
