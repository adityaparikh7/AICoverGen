"""
Separation model catalogs and unified wrapper for the audio-separator library.

This module defines available models for each separation task and provides
a consistent `separate_audio()` function used by the pipeline and UIs.
"""

import gc
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEPARATION_MODELS_DIR = os.path.join(BASE_DIR, 'separation_models')

# Ensure the models directory exists
os.makedirs(SEPARATION_MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model catalogs: display name → model filename
# The first entry in each dict is the default (best quality).
# ---------------------------------------------------------------------------

VOCAL_INSTRUMENTAL_MODELS = {
    "BS-Roformer (Best Quality)": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "MDX23C HQ v2 (Very Good)": "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
    "MDX23C HQ (Good)": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "UVR-MDX-NET Voc FT (Legacy)": "UVR-MDX-NET-Voc_FT.onnx",
}

KARAOKE_MODELS = {
    "Mel-Roformer Karaoke (Best)": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "UVR MDX Karaoke 2 (Legacy)": "UVR_MDXNET_KARA_2.onnx",
}

DEREVERB_MODELS = {
    "UVR DeEcho-DeReverb (Best)": "UVR-DeEcho-DeReverb.pth",
    "Reverb HQ By FoxJoy (Legacy)": "Reverb_HQ_By_FoxJoy.onnx",
    "None (Skip)": None,
}

INSTRUMENTAL_STEM_MODELS = {
    "HTDemucs 6-Stem (Best)": "htdemucs_6s.yaml",
    "HTDemucs Fine-Tuned 4-Stem": "htdemucs_ft.yaml",
    "HTDemucs Standard 4-Stem (Fast)": "htdemucs.yaml",
}

# Stem names output by each model type
STEM_NAMES_6S = ["drums", "bass", "guitar", "piano", "other"]
STEM_NAMES_4S = ["drums", "bass", "other"]

# Convenience: default display-name keys
DEFAULT_VOCAL_MODEL = list(VOCAL_INSTRUMENTAL_MODELS.keys())[0]
DEFAULT_KARAOKE_MODEL = list(KARAOKE_MODELS.keys())[0]
DEFAULT_DEREVERB_MODEL = list(DEREVERB_MODELS.keys())[0]
DEFAULT_STEM_MODEL = list(INSTRUMENTAL_STEM_MODELS.keys())[0]


def _resolve_model_filename(display_name, catalog):
    """Convert a UI display name to an actual model filename."""
    filename = catalog.get(display_name)
    if filename is None and display_name != "None (Skip)":
        # If the display name isn't found, treat it as a raw filename
        filename = display_name
    return filename


def _base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _stem_of(path):
    """
    Return the lowercased stem name from an audio-separator default output name,
    <input>_(<Stem>)_<model>.wav. Only the last token counts, because the input
    name can carry tokens from earlier separation steps.
    """
    match = re.search(r'_\(([^()]+)\)_[^()]*$', _base_name(path))
    return match.group(1).lower() if match else ''


def _separate_named(input_path, output_dir, model_filename, denoise, base_name, stem_suffixes):
    """
    Run a separation that writes each stem as <base_name>_<suffix>.wav.

    stem_suffixes maps audio-separator stem names (case-insensitive) to suffixes.
    Outputs are identified by those suffixes, never by substrings of the song title
    or model name. Returns ({suffix: path}, [outputs whose stem wasn't in the map]).
    """
    custom_output_names = {stem: f'{base_name}_{suffix}' for stem, suffix in stem_suffixes.items()}
    output_files = separate_audio(input_path, output_dir, model_filename, denoise, custom_output_names)

    named, unmatched = {}, []
    for f in output_files:
        name = _base_name(f)
        suffix = next((s for s in stem_suffixes.values() if name.endswith(f'_{s}')), None)
        if suffix:
            named[suffix] = f
        else:
            unmatched.append(f)
    return named, unmatched


def _take_unmatched(unmatched, prefer=lambda stem: False):
    """
    Take an output whose stem the custom names didn't cover (a model outside the
    catalogs), preferring one whose stem name satisfies `prefer`.
    """
    if not unmatched:
        return None
    pick = next((f for f in unmatched if prefer(_stem_of(f))), unmatched[0])
    unmatched.remove(pick)
    return pick


def separate_audio(input_path, output_dir, model_filename, denoise=True, custom_output_names=None):
    """
    Separate audio using the audio-separator library.

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory to write output stems.
        model_filename: The model filename (e.g., "model_bs_roformer_ep_317_sdr_12.9755.ckpt").
        denoise: Whether to enable denoising (phase-inversion trick for MDX-Net models).
        custom_output_names: Optional {stem name: output file name without extension}.

    Returns:
        list: Paths to output stem files, in the order the model architecture writes them.
    """
    from audio_separator.separator import Separator

    os.makedirs(output_dir, exist_ok=True)

    separator = Separator(
        model_file_dir=SEPARATION_MODELS_DIR,
        output_dir=output_dir,
        output_format="WAV",
        normalization_threshold=0.9,
        # denoise_enabled=denoise,
        output_single_stem=None,
        sample_rate=44100,
    )

    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(input_path, custom_output_names=custom_output_names)

    # Convert to absolute paths since audio-separator returns just filenames
    output_files = [os.path.join(output_dir, f) if not os.path.isabs(f) else f for f in output_files]

    # Clean up to free memory
    del separator
    gc.collect()

    return output_files


# Output names below are <base_name>_<suffix>.wav. main.get_audio_paths finds cached
# files by these suffixes, so keep the two in sync.

def separate_vocals_instrumental(input_path, output_dir, model_display_name=None, denoise=True, base_name=None):
    """
    Separate vocals from instrumentals into <base>_Vocals.wav and <base>_Instrumental.wav.

    Returns:
        (vocals_path, instrumentals_path)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_VOCAL_MODEL
    model_filename = _resolve_model_filename(model_display_name, VOCAL_INSTRUMENTAL_MODELS)

    named, unmatched = _separate_named(
        input_path, output_dir, model_filename, denoise, base_name or _base_name(input_path),
        {'Vocals': 'Vocals', 'Instrumental': 'Instrumental'},
    )
    vocals_path = named.get('Vocals') or _take_unmatched(
        unmatched, lambda stem: 'vocal' in stem and not stem.startswith('no '))
    instrumentals_path = named.get('Instrumental') or _take_unmatched(unmatched)

    return vocals_path, instrumentals_path


def separate_main_backup_vocals(vocals_path, output_dir, model_display_name=None, denoise=True, base_name=None):
    """
    Separate main vocals from backup vocals (karaoke split) into
    <base>_Vocals_Main.wav and <base>_Vocals_Backup.wav.

    Returns:
        (main_vocals_path, backup_vocals_path)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_KARAOKE_MODEL
    model_filename = _resolve_model_filename(model_display_name, KARAOKE_MODELS)

    # Karaoke models output the lead as "Vocals" and everything else (the backing
    # vocals, since the input is vocals only) as "Instrumental".
    named, unmatched = _separate_named(
        vocals_path, output_dir, model_filename, denoise, base_name or _base_name(vocals_path),
        {'Vocals': 'Vocals_Main', 'Instrumental': 'Vocals_Backup'},
    )
    main_vocals_path = named.get('Vocals_Main') or _take_unmatched(unmatched, lambda stem: 'lead' in stem)
    backup_vocals_path = named.get('Vocals_Backup') or _take_unmatched(unmatched)

    return main_vocals_path, backup_vocals_path


def apply_dereverb(vocals_path, output_dir, model_display_name=None, denoise=True, base_name=None):
    """
    Remove reverb/echo from vocals. Writes the dry signal to <base>_Vocals_Main_DeReverb.wav
    and the removed reverb to <base>_Vocals_Main_Reverb.wav.

    Returns:
        dereverbed_vocals_path (or the original path if model is None/skipped)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_DEREVERB_MODEL

    model_filename = _resolve_model_filename(model_display_name, DEREVERB_MODELS)
    if model_filename is None:
        # "None (Skip)" was selected
        return vocals_path

    named, unmatched = _separate_named(
        vocals_path, output_dir, model_filename, denoise, base_name or _base_name(vocals_path),
        {
            'No Reverb': 'Vocals_Main_DeReverb', 'No Echo': 'Vocals_Main_DeReverb',
            'Reverb': 'Vocals_Main_Reverb', 'Echo': 'Vocals_Main_Reverb',
        },
    )
    return named.get('Vocals_Main_DeReverb') or _take_unmatched(
        unmatched, lambda stem: stem.startswith('no ') or 'dry' in stem)


def separate_instrumental_stems(instrumental_path, output_dir, model_display_name=None, base_name=None):
    """
    Separate an instrumental track into individual instrument stems.

    Uses Demucs models via audio-separator to split the instrumental into
    drums, bass, guitar, piano, other, and vocal remnants, written as
    <base>_Stem_<Name>.wav and <base>_Stem_Vocal_Remnants.wav.

    Args:
        instrumental_path: Path to the instrumental WAV file.
        output_dir: Directory to write output stem files.
        model_display_name: Display name from INSTRUMENTAL_STEM_MODELS catalog.
        base_name: Prefix for output file names (default: the input file name).

    Returns:
        dict: Mapping of stem name to file path, e.g.
              {"drums": "/path/drums.wav", "bass": ..., "vocal_remnants": ...}
    """
    if model_display_name is None:
        model_display_name = DEFAULT_STEM_MODEL
    model_filename = _resolve_model_filename(model_display_name, INSTRUMENTAL_STEM_MODELS)

    # audio-separator stem name -> (key in the returned dict, output file suffix)
    stems = {
        'Drums': ('drums', 'Stem_Drums'),
        'Bass': ('bass', 'Stem_Bass'),
        'Guitar': ('guitar', 'Stem_Guitar'),
        'Piano': ('piano', 'Stem_Piano'),
        'Other': ('other', 'Stem_Other'),
        'Vocals': ('vocal_remnants', 'Stem_Vocal_Remnants'),
    }
    named, unmatched = _separate_named(
        instrumental_path, output_dir, model_filename, True, base_name or _base_name(instrumental_path),
        {stem: suffix for stem, (_, suffix) in stems.items()},
    )
    stem_paths = {key: named[suffix] for key, suffix in stems.values() if suffix in named}

    # Assign any unmatched files to 'other' if not already set
    if 'other' not in stem_paths and unmatched:
        stem_paths['other'] = unmatched.pop(0)

    return stem_paths
