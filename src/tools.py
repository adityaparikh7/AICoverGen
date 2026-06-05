"""
Unified Tools UI — Audio Splitter + Voice Conversion

Combines the functionality of the old split_audio.py and voice_only.py
into a single tabbed Gradio application.

Usage:
    python src/tools.py
"""

import os
import sys

# Ensure PyTorch falls back to CPU for unsupported MPS operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import torch

from rvc import Config, load_hubert, get_vc, rvc_infer
from separator_models import (
    VOCAL_INSTRUMENTAL_MODELS,
    KARAOKE_MODELS,
    DEREVERB_MODELS,
    DEFAULT_VOCAL_MODEL,
    DEFAULT_KARAOKE_MODEL,
    DEFAULT_DEREVERB_MODEL,
    separate_vocals_instrumental,
    separate_main_backup_vocals,
    apply_dereverb,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


# ---------------------------------------------------------------------------
# Audio Splitter helpers
# ---------------------------------------------------------------------------

def split_audio_webui(input_audio, vocal_model, karaoke_model, dereverb_model):
    """Split audio into instrumental, backup vocals, main vocals, and de-reverbed main vocals."""
    if input_audio is None:
        raise gr.Error("Please upload an audio file.")

    input_path = input_audio
    if hasattr(input_audio, "name"):
        input_path = input_audio.name

    # Prepare output folder
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_folder = os.path.join(output_dir, f"{base_name}_audio_split")
    os.makedirs(out_folder, exist_ok=True)

    # Step 1: Vocals / Instrumental separation
    vocals_path, instrumentals_path = separate_vocals_instrumental(
        input_path, out_folder, model_display_name=vocal_model
    )

    # Step 2: Main / Backup vocals split
    main_vocals_path, backup_vocals_path = separate_main_backup_vocals(
        vocals_path, out_folder, model_display_name=karaoke_model
    )

    # Step 3: De-reverb on main vocals
    main_vocals_dereverb_path = apply_dereverb(
        main_vocals_path, out_folder, model_display_name=dereverb_model
    )

    return instrumentals_path, backup_vocals_path, main_vocals_path, main_vocals_dereverb_path


# ---------------------------------------------------------------------------
# Voice Conversion helpers
# ---------------------------------------------------------------------------

def get_current_models(models_dir):
    """List available RVC voice models."""
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [
        item for item in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, item)) and item not in items_to_remove
    ]


def refresh_models():
    """Refresh the voice model dropdown."""
    new_models = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=new_models)


def get_rvc_model(voice_model):
    """Find the .pth and .index files for a voice model."""
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        raise gr.Error(f'No model file exists in {model_dir}.')

    return (
        os.path.join(model_dir, rvc_model_filename),
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''
    )


def simple_voice_conversion(input_audio, voice_model, pitch_change, index_rate,
                            filter_radius, rms_mix_rate, protect, f0_method, crepe_hop_length):
    """Convert voice using RVC."""
    if not input_audio or not voice_model:
        raise gr.Error('Please provide an audio file and select a voice model.')

    # Get model paths
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)

    # Device selection — Mac uses CPU + float32 for reliability
    if torch.cuda.is_available():
        device = "cuda:0"
        is_half = True
    else:
        device = "cpu"
        is_half = False

    config = Config(device, is_half)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    # Output path
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_filename = f"{base_name}_to_{voice_model}_p{pitch_change}.wav"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Convert voice (pitch_change is in octaves, convert to semitones)
    pitch_change_semitones = pitch_change * 12
    rvc_infer(
        rvc_index_path, index_rate, input_audio, output_path,
        pitch_change_semitones, f0_method, cpt, version, net_g,
        filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length,
        vc, hubert_model
    )

    return output_path


# ---------------------------------------------------------------------------
# UI visibility helpers
# ---------------------------------------------------------------------------

def show_crepe_slider(method):
    return gr.update(visible=method == 'mangio-crepe')


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

voice_models = get_current_models(rvc_models_dir)

with gr.Blocks(title='AI Audio Tools') as app:
    gr.Markdown('# 🎵 AI Audio Tools')
    gr.Markdown('Split audio into stems or convert voices using AI models.')

    # === Tab 1: Audio Splitter ===
    with gr.Tab("🎤 Audio Splitter"):
        gr.Markdown('Upload a song and split it into vocals and instrumentals using AI separation models.')

        with gr.Row():
            audio_in = gr.Audio(label="Upload Audio", type="filepath")

        with gr.Accordion("Separation Model Selection", open=True):
            gr.Markdown(
                'Choose which AI models to use for each separation step. '
                'Better models produce cleaner stems but take longer. '
                'Models are auto-downloaded on first use.'
            )
            with gr.Row():
                vocal_model_dropdown = gr.Dropdown(
                    choices=list(VOCAL_INSTRUMENTAL_MODELS.keys()),
                    value=DEFAULT_VOCAL_MODEL,
                    label='Vocal / Instrumental Model',
                    info='Separates vocals from instrumentals. BS-Roformer is state-of-the-art.'
                )
                karaoke_model_dropdown = gr.Dropdown(
                    choices=list(KARAOKE_MODELS.keys()),
                    value=DEFAULT_KARAOKE_MODEL,
                    label='Main / Backup Vocals Model',
                    info='Separates lead vocals from backing vocals.'
                )
                dereverb_model_dropdown = gr.Dropdown(
                    choices=list(DEREVERB_MODELS.keys()),
                    value=DEFAULT_DEREVERB_MODEL,
                    label='De-Reverb Model',
                    info='Removes reverb/echo from vocals. Select "None" to skip.'
                )

        with gr.Row():
            split_btn = gr.Button("Split Audio", variant="primary")

        with gr.Row():
            instrumental_out = gr.Audio(label="Instrumental", interactive=False)
            backup_vocals_out = gr.Audio(label="Backup Vocals", interactive=False)
        with gr.Row():
            main_vocals_out = gr.Audio(label="Main Vocals", interactive=False)
            main_vocals_dereverb_out = gr.Audio(label="Main Vocals (De-Reverbed)", interactive=False)

        split_btn.click(
            split_audio_webui,
            inputs=[audio_in, vocal_model_dropdown, karaoke_model_dropdown, dereverb_model_dropdown],
            outputs=[instrumental_out, backup_vocals_out, main_vocals_out, main_vocals_dereverb_out]
        )

    # === Tab 2: Voice Conversion ===
    with gr.Tab("🎙️ Voice Conversion"):
        gr.Markdown('Upload an audio file and convert it directly using RVC voice models — no audio splitting.')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    voice_model = gr.Dropdown(
                        voice_models,
                        label='Voice Model',
                        info='Select a voice model from rvc_models folder'
                    )
                    refresh_btn = gr.Button('🔄', size='sm', min_width=40)
                vc_audio_in = gr.Audio(label='Upload Audio File', type='filepath')

            with gr.Column():
                pitch_change = gr.Slider(
                    -3, 3, value=0, step=1,
                    label='Pitch Change (Octaves)',
                    info='1 for male→female, -1 for female→male'
                )
                index_rate = gr.Slider(
                    0, 1, value=0.5,
                    label='Index Rate',
                    info="Controls how much of the AI voice's accent to keep"
                )
                filter_radius = gr.Slider(
                    0, 7, value=3, step=1,
                    label='Filter Radius',
                    info='If >=3: apply median filtering to reduce breathiness'
                )

            with gr.Column():
                rms_mix_rate = gr.Slider(
                    0, 1, value=0.25,
                    label='RMS Mix Rate',
                    info="Control loudness mixing"
                )
                protect = gr.Slider(
                    0, 0.5, value=0.33,
                    label='Protect Rate',
                    info='Protect consonants and breath sounds'
                )
                f0_method = gr.Dropdown(
                    ['rmvpe', 'mangio-crepe'],
                    value='rmvpe',
                    label='Pitch Detection',
                    info='rmvpe = clarity, mangio-crepe = smoother'
                )
                crepe_hop_length = gr.Slider(
                    32, 320, value=128, step=1,
                    label='Crepe Hop Length',
                    visible=False
                )

        f0_method.change(show_crepe_slider, inputs=f0_method, outputs=crepe_hop_length)
        refresh_btn.click(refresh_models, outputs=voice_model)

        convert_btn = gr.Button('Convert Voice', variant='primary')
        vc_output_audio = gr.Audio(label='Converted Audio')

        convert_btn.click(
            simple_voice_conversion,
            inputs=[
                vc_audio_in, voice_model, pitch_change, index_rate,
                filter_radius, rms_mix_rate, protect, f0_method, crepe_hop_length
            ],
            outputs=vc_output_audio
        )


if __name__ == '__main__':
    app.launch()
