import torch
import torchaudio
import gradio as gr
from os import getenv
import os
import tempfile
from pydub import AudioSegment
import io

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import io

device = "cuda"
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

CHUNK_SIZE = 50 # ~40 fit in 1500 tokens, so the cap for 2500 is probably 50ish
MIN_CHUNK = 10
OVERLAP_CHUNKS = False

def wav_to_mp3(wav_data: np.ndarray, sample_rate: int, output_filename: str) -> str:
    """Convert numpy wav data to MP3 file and return the path"""
    # Convert to int16 PCM
    wav_data = (wav_data * 32767).astype(np.int16)
    
    # Create WAV in memory
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, wav_data)
    wav_io.seek(0)
    
    # Convert to MP3
    audio_segment = AudioSegment.from_wav(wav_io)
    
    # Create temporary file for MP3
    audio_segment.export(output_filename, format='mp3')
    return output_filename


def get_speaker_files():
    speaker_dir = "speakers"
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)
    speaker_files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith(('.wav', '.mp3'))]
    return speaker_files


def split_last_seconds(wav: torch.Tensor, sr: int, duration: float = .2) -> tuple[torch.Tensor, torch.Tensor]:
    """Split audio tensor into (prefix, last_second)"""
    last_second_samples = int(sr*duration)  # Since sr is samples per second

    if wav.size(-1) <= last_second_samples:
        return wav, None
    
    most_of_wav = wav[..., :-last_second_samples]
    last_second = wav[..., -last_second_samples:]
    return most_of_wav, last_second


def add_silence_padding(wav: torch.Tensor, sr: int, duration: float = 0.1) -> torch.Tensor:
    """Add small silence padding to the audio."""
    silence_samples = int(sr * duration)
    silence = torch.zeros((*wav.shape[:-1], silence_samples), device=wav.device)
    return torch.cat([wav, silence], dim=-1)


def is_audio_too_short(wav_tensor, min_samples=1024):  # adjust min_samples as needed
    return wav_tensor.size(-1) < min_samples


def split_into_chunks(text: str, max_words: int = 20) -> list[str]:
    """
    Split text into chunks of up to max_words, preferring natural break points.
    Tries to avoid splitting phrases awkwardly.

    Args:
        text: The input text to split
        max_words: Maximum number of words per chunk

    Returns:
        List of text chunks
    """
    # Define break points in order of preference
    major_breaks = ['. ', '! ', '? ']  # Sentence endings
    minor_breaks = ['; ', ': ', ', ']  # Clause breaks

    chunks = []
    while text:
        text = text.strip()

        # If remaining text is short enough, add it as the final chunk
        if len(text.split()) <= max_words:
            if text:
                chunks.append(text)
            break

        # Look for break points within a window slightly larger than max_words
        # This allows us to look a bit ahead for better break points
        search_window = ' '.join(text.split()[:max_words + 5])

        # First try to find major breaks within the normal word limit
        normal_window = ' '.join(text.split()[:max_words])
        best_pos = -1
        best_break = None

        # Try major breaks first within normal window
        for break_point in major_breaks:
            pos = normal_window.rfind(break_point)

            if pos > best_pos and pos > MIN_CHUNK:
                best_pos = pos
                best_break = break_point

        # If no major break found, try minor breaks within normal window
        if best_pos == -1:
            for break_point in minor_breaks:
                pos = normal_window.rfind(break_point)
                if pos > best_pos and pos > MIN_CHUNK:
                    best_pos = pos
                    best_break = break_point

        # If still no break found within normal window, look in extended window
        if best_pos == -1:
            for break_point in (major_breaks + minor_breaks):
                pos = search_window.rfind(break_point)
                if best_pos < pos and pos < len(normal_window) * 1.2 and pos > MIN_CHUNK:  # Allow slight overflow
                    best_pos = pos
                    best_break = break_point

        # If still no break found, force break at word boundary near max_words
        if best_pos == -1:
            words = text.split()[:max_words]
            best_pos = len(' '.join(words))
            chunks.append(text[:best_pos].strip())
            text = text[best_pos:].strip()
        else:
            # Include the break point in the chunk
            chunks.append(text[:best_pos + len(best_break)].strip())
            text = text[best_pos + len(best_break):].strip()

    return chunks


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio_choice,
    speaker_audio_upload,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    min_p,
    seed,
    randomize_seed,
    unconditional_keys,
    txt_files,
    progress=gr.Progress(),
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    Handles both single text input and text file input.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_audio = speaker_audio_choice if speaker_audio_choice else speaker_audio_upload
    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    min_p = float(min_p)
    seed = int(seed)
    max_new_tokens = 86 * 30

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)

    speaker_embedding = None
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        wav, sr = torchaudio.load(speaker_audio)
        speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)
    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    if txt_files:
        print("Processing text files...")
        output_files = []
        
        for file in txt_files:
            output_filename = os.path.splitext(file.name)[0] + '.mp3'
            output_filename = os.path.basename(output_filename)
            print(f"Processing {file.name} -> {output_filename}")
            
            all_wavs = []
            sr_out_final = None
            previous_wav = None
            previous_text = None

            # Initial prefix audio
            initial_prefix = prefix_audio if prefix_audio else None

            with open(file.name, 'r', encoding='utf-8') as f:
                full_text = f.read()

            chunks = split_into_chunks(full_text, max_words=CHUNK_SIZE)
            
            for i, chunk in enumerate(chunks):
                print(f"Generating audio for chunk {i+1}/{len(chunks)}: '{chunk}'")
                
                # Handle prefix audio
                audio_prefix_codes = None
                prefix_length = None
                
                if i == 0 and initial_prefix is not None:
                    # Handle initial prefix from file
                    wav_prefix, sr_prefix = torchaudio.load(initial_prefix)
                    wav_prefix = wav_prefix.mean(0, keepdim=True)
                    wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
                    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
                    with torch.autocast(device, dtype=torch.float32):
                        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))
                    prefix_length = wav_prefix.size(-1)
                elif i > 0 and previous_wav is not None:
                    # Check if audio is long enough before using as prefix
                    if not is_audio_too_short(previous_wav) and OVERLAP_CHUNKS:
                        previous_wav = previous_wav.to(device, dtype=torch.float32)
                        with torch.autocast(device, dtype=torch.float32):
                            audio_prefix_codes = selected_model.autoencoder.encode(previous_wav.unsqueeze(0))
                        prefix_length = previous_wav.size(-1)
                    else:
                        # Skip using prefix for this chunk
                        audio_prefix_codes = None
                        prefix_length = None
                        previous_text = None

                combined_text = chunk
                if previous_text:
                    combined_text = previous_text + " " + chunk

                # Generate current chunk
                cond_dict = make_cond_dict(
                    text=combined_text,
                    language=language,
                    speaker=speaker_embedding,
                    emotion=emotion_tensor,
                    vqscore_8=vq_tensor,
                    fmax=fmax,
                    pitch_std=pitch_std,
                    speaking_rate=speaking_rate,
                    dnsmos_ovrl=dnsmos_ovrl,
                    speaker_noised=speaker_noised_bool,
                    device=device,
                    unconditional_keys=unconditional_keys,
                )
                conditioning = selected_model.prepare_conditioning(cond_dict)

                estimated_generation_duration = 30 * len(combined_text) / 400
                estimated_total_steps = int(estimated_generation_duration * 86)

                def update_progress_file(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
                    progress((step, estimated_total_steps))
                    return True

                codes = selected_model.generate(
                    prefix_conditioning=conditioning,
                    audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    batch_size=1,
                    sampling_params=dict(min_p=min_p),
                    callback=update_progress_file,
                )

                wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
                sr_out = selected_model.autoencoder.sampling_rate
                sr_out_final = sr_out

                wav_out_2d = wav_out.squeeze(0)
                if wav_out_2d.dim() == 2 and wav_out_2d.size(0) > 1:
                    wav_out_2d = wav_out_2d[0:1, :]

                if prefix_length is not None:
                    wav_out_2d = wav_out_2d[..., prefix_length:]
                    prefix_length = None

                all_wavs.append(wav_out_2d)
                previous_wav = wav_out_2d
                previous_text = chunk

            if all_wavs and sr_out_final is not None:
                concatenated_wav = torch.cat(all_wavs, dim=-1)
                    
                wav_numpy = concatenated_wav.squeeze().numpy()
                wav_data = (wav_numpy * 32767).astype(np.int16)
                wav_io = io.BytesIO()
                wavfile.write(wav_io, sr_out_final, wav_data)
                wav_io.seek(0)
                
                audio_segment = AudioSegment.from_wav(wav_io)
                audio_segment.export(output_filename, format='mp3')
                output_files.append(output_filename)
                print(f"Saved to {output_filename}")
        
        if output_files:
            audio_segment = AudioSegment.from_mp3(output_files[-1])
            numpy_array = np.array(audio_segment.get_array_of_samples())
            return (sr_out_final, numpy_array), seed
        else:
            return (None, None), seed

    else: # Original single text generation
        print("Processing single text input...")
        audio_prefix_codes = None
        if prefix_audio is not None:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            with torch.autocast(device, dtype=torch.float32):
                audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))


        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised_bool,
            device=device,
            unconditional_keys=unconditional_keys,
        )
        conditioning = selected_model.prepare_conditioning(cond_dict)

        estimated_generation_duration = 30 * len(text) / 400
        estimated_total_steps = int(estimated_generation_duration * 86)

        def update_progress_single(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
            progress((step, estimated_total_steps))
            return True

        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=dict(min_p=min_p),
            callback=update_progress_single,
        )

        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        
        # Save as MP3
        output_filename = "output.mp3"
        wav_numpy = wav_out.squeeze().numpy()
        wav_data = (wav_numpy * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        wavfile.write(wav_io, sr_out, wav_data)
        wav_io.seek(0)
        
        audio_segment = AudioSegment.from_wav(wav_io)
        audio_segment.export(output_filename, format='mp3')
        print(f"Saved to {output_filename}")
        
        # Load the MP3 for preview
        audio_segment = AudioSegment.from_mp3(output_filename)
        numpy_array = np.array(audio_segment.get_array_of_samples())
        
        return (sr_out, numpy_array), seed



def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"],
                    value="Zyphra/Zonos-v0.1-transformer",
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                    max_length=500,  # approximately
                    visible=True # Initially visible for text input
                )
                txt_files = gr.File(
                    label="Input Text Files", 
                    file_types=['.txt'], 
                    visible=True, 
                    file_count="multiple"
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Dropdown(
                    choices=[""] + get_speaker_files(),
                    value="",
                    label="Speaker Audio (select from speakers folder or upload below)",
                    type="value",
                )
                speaker_audio_upload = gr.Audio(
                    label="Optional Speaker Audio Upload",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=12.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                min_p_slider = gr.Slider(0.0, 1.0, 0.10, 0.01, label="Min P")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                speaker_audio_upload,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                min_p_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
                txt_files,
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
