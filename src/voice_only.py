import os
import gradio as gr
import torch
from rvc import Config, load_hubert, get_vc, rvc_infer

# Ensure PyTorch falls back to CPU for unsupported MPS operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_current_models(models_dir):
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, item)) and item not in items_to_remove]

voice_models = get_current_models(rvc_models_dir)

def refresh_models():
    new_models = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=new_models)

def get_rvc_model(voice_model):
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

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''

def simple_voice_conversion(input_audio, voice_model, pitch_change, index_rate, filter_radius, rms_mix_rate, protect, f0_method, crepe_hop_length):
    if not input_audio or not voice_model:
        raise gr.Error('Please provide an audio file and select a voice model.')

    # Get model paths
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)

    # Device selection — Mac uses CPU + float32 for reliability.
    # MPS has known issues with fairseq HuBERT and RVC model ops.
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

    # Convert voice
    pitch_change_semitones = pitch_change * 12  # Convert octaves to semitones
    rvc_infer(rvc_index_path, index_rate, input_audio, output_path, pitch_change_semitones, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)

    return output_path

with gr.Blocks(title='Simple AI Voice Conversion') as app:
    gr.Markdown('# Simple AI Voice Conversion')
    gr.Markdown('Upload an audio file and convert it directly using RVC models - no audio splitting.')

    with gr.Row():
        with gr.Column():
            with gr.Row():
                voice_model = gr.Dropdown(voice_models, label='Voice Model', info='Select a voice model from rvc_models folder')
                refresh_btn = gr.Button('🔄', size='sm', min_width=40)
            input_audio = gr.Audio(label='Upload Audio File', type='filepath')

        with gr.Column():
            pitch_change = gr.Slider(-3, 3, value=0, step=1, label='Pitch Change (Octaves)', info='1 for male→female, -1 for female→male')
            index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to keep")
            filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='If >=3: apply median filtering to reduce breathiness')

        with gr.Column():
            rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate', info="Control loudness mixing")
            protect = gr.Slider(0, 0.5, value=0.33, label='Protect Rate', info='Protect consonants and breath sounds')
            f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Pitch Detection', info='rmvpe = clarity, mangio-crepe = smoother')
            crepe_hop_length = gr.Slider(32, 320, value=128, step=1, label='Crepe Hop Length', visible=False)

    def show_crepe_slider(method):
        return gr.update(visible=method == 'mangio-crepe')

    f0_method.change(show_crepe_slider, inputs=f0_method, outputs=crepe_hop_length)
    refresh_btn.click(refresh_models, outputs=voice_model)

    convert_btn = gr.Button('Convert Voice', variant='primary')
    output_audio = gr.Audio(label='Converted Audio')

    convert_btn.click(
        simple_voice_conversion,
        inputs=[input_audio, voice_model, pitch_change, index_rate, filter_radius, rms_mix_rate, protect, f0_method, crepe_hop_length],
        outputs=output_audio
    )

if __name__ == '__main__':
    app.launch()