import torch
import torchaudio
import gradio as gr
from os import getenv
import os
import tempfile

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

device = "cuda"
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None


def get_speaker_files():
    speaker_dir = "speakers"
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)
    speaker_files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith(('.wav', '.mp3'))]
    return speaker_files


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
    txt_file,
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

    if txt_file:
        print("Processing text file...")
        concatenated_wav = None
        sr_out_final = None
        
        # Initial prefix audio
        initial_prefix = prefix_audio if prefix_audio else None

        with open(txt_file.name, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]  # Get all non-empty lines
            
            for i, current_text in enumerate(lines):
                print(f"Generating audio for line {i+1}/{len(lines)}: '{current_text}'")

                # Use either initial prefix or last generated audio as prefix
                audio_prefix_codes = None
                if i == 0 and initial_prefix is not None:
                    wav_prefix, sr_prefix = torchaudio.load(initial_prefix)
                    wav_prefix = wav_prefix.mean(0, keepdim=True)

                    wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
                    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
                    with torch.autocast(device, dtype=torch.float32):
                        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

                cond_dict = make_cond_dict(
                    text=current_text,
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

                estimated_generation_duration = 30 * len(current_text) / 400
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

                wav_out_2d = wav_out.squeeze(0)
                if wav_out_2d.dim() == 2 and wav_out_2d.size(0) > 1:
                    wav_out_2d = wav_out_2d[0:1, :]
                
                # Add small silence between lines
                silence_samples = int(0.1 * sr_out)  # 200ms silence
                silence = torch.zeros((1, silence_samples), device=wav_out_2d.device)
                wav_out_2d = torch.cat([wav_out_2d, silence], dim=-1)

                if concatenated_wav is None:
                    concatenated_wav = wav_out_2d
                else:
                    concatenated_wav = torch.cat([concatenated_wav, wav_out_2d], dim=-1)
                sr_out_final = sr_out

        if concatenated_wav is not None and sr_out_final is not None:
            return (sr_out_final, concatenated_wav.squeeze().numpy()), seed
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
        return (sr_out, wav_out.squeeze().numpy()), seed



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
                txt_file = gr.File(label="Input Text File", file_types=['.txt'], visible=True) # Initially visible
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
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                min_p_slider = gr.Slider(0.0, 1.0, 0.15, 0.01, label="Min P")
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
                txt_file, # Added txt_file input
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
