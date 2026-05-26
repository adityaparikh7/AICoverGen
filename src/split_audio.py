import os
import json
import shutil
import gradio as gr

from mdx import run_mdx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def split_audio_webui(input_audio):
    if input_audio is None:
        raise gr.Error("Please upload an audio file.")

    # Save uploaded file to disk if needed
    input_path = input_audio
    if hasattr(input_audio, "name"):
        input_path = input_audio.name

    # Prepare output folder
    out_folder = os.path.join(output_dir, "audio_split_" + os.path.splitext(os.path.basename(input_path))[0])
    os.makedirs(out_folder, exist_ok=True)

    # Load MDX model params
    with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
        mdx_model_params = json.load(infile)

    # 1. Split into vocals and instrumentals
    model_voc_inst = os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx')
    vocals_path, instrumentals_path = run_mdx(
        mdx_model_params,
        out_folder,
        model_voc_inst,
        input_audio,
        denoise=True,
        keep_orig=True
    )

    # 2. Split vocals into main and backup vocals
    model_kara = os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx')
    backup_vocals_path, main_vocals_path = run_mdx(
        mdx_model_params,
        out_folder,
        model_kara,
        vocals_path,
        suffix='Backup',
        invert_suffix='Main',
        denoise=True
    )

    # 3. Remove reverb from main vocals (optional)
    # model_dereverb = os.path.join(mdxnet_models_dir, 'UVR_MDXNET_DEREVERB.onnx')
    # dereverb_main_vocals_path, _ = run_mdx(
    #     mdx_model_params,
    #     out_folder,
    #     model_dereverb,
    #     main_vocals_path,
    #     suffix='Dereverb',
    #     denoise=True
    # )



    # Return for download
    return instrumentals_path, backup_vocals_path, main_vocals_path

with gr.Blocks(title="Audio Splitter: Vocals & Instrumentals") as app:
    gr.Markdown("# 🎤 Audio Splitter")
    gr.Markdown("Upload a song and split it into vocals and instrumentals using AI.")

    with gr.Row():
        audio_in = gr.Audio(label="Upload Audio", type="filepath")
    with gr.Row():
        split_btn = gr.Button("Split Audio")
    with gr.Row():
        instrumental_out = gr.Audio(label="Instrumental", interactive=False)
        backup_vocals_out = gr.Audio(label="Backup Vocals", interactive=False)
        main_vocals_out = gr.Audio(label="Main Vocals", interactive=False)

    split_btn.click(
        split_audio_webui,
        inputs=audio_in,
        outputs=[instrumental_out, backup_vocals_out, main_vocals_out]
    )

app.launch()