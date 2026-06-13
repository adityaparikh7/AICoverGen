"""
Separation model catalogs and unified wrapper for the audio-separator library.

This module defines available models for each separation task and provides
a consistent `separate_audio()` function used by the pipeline and UIs.
"""

import gc
import os

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


def separate_audio(input_path, output_dir, model_filename, denoise=True):
    """
    Separate audio using the audio-separator library.

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory to write output stems.
        model_filename: The model filename (e.g., "model_bs_roformer_ep_317_sdr_12.9755.ckpt").
        denoise: Whether to enable denoising (phase-inversion trick for MDX-Net models).

    Returns:
        list: Paths to output stem files [primary_stem, secondary_stem].
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
    output_files = separator.separate(input_path)

    # Convert to absolute paths since audio-separator returns just filenames
    output_files = [os.path.join(output_dir, f) if not os.path.isabs(f) else f for f in output_files]

    # Clean up to free memory
    del separator
    gc.collect()

    return output_files


def separate_vocals_instrumental(input_path, output_dir, model_display_name=None, denoise=True):
    """
    Separate vocals from instrumentals.

    Returns:
        (vocals_path, instrumentals_path)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_VOCAL_MODEL
    model_filename = _resolve_model_filename(model_display_name, VOCAL_INSTRUMENTAL_MODELS)

    output_files = separate_audio(input_path, output_dir, model_filename, denoise)

    # audio-separator returns [primary, secondary] — order depends on model.
    # For vocal models, primary is typically Vocals, secondary is Instrumental
    # But we need to detect which is which based on naming.
    vocals_path = None
    instrumentals_path = None
    for f in output_files:
        f_lower = f.lower()
        if 'vocal' in f_lower:
            vocals_path = f
        elif 'instrument' in f_lower or 'no_vocal' in f_lower:
            instrumentals_path = f

    # Fallback: if naming detection fails, assume first=primary (Vocals), second=secondary
    if vocals_path is None and instrumentals_path is None:
        vocals_path = output_files[0] if len(output_files) > 0 else None
        instrumentals_path = output_files[1] if len(output_files) > 1 else None
    elif vocals_path is None:
        vocals_path = [f for f in output_files if f != instrumentals_path][0] if len(output_files) > 1 else None
    elif instrumentals_path is None:
        instrumentals_path = [f for f in output_files if f != vocals_path][0] if len(output_files) > 1 else None

    return vocals_path, instrumentals_path


def separate_main_backup_vocals(vocals_path, output_dir, model_display_name=None, denoise=True):
    """
    Separate main vocals from backup vocals (karaoke split).

    Returns:
        (main_vocals_path, backup_vocals_path)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_KARAOKE_MODEL
    model_filename = _resolve_model_filename(model_display_name, KARAOKE_MODELS)

    output_files = separate_audio(vocals_path, output_dir, model_filename, denoise)

    # For karaoke models: primary is typically the "karaoke" (backing/no-lead),
    # secondary is the lead/main vocals
    main_vocals_path = None
    backup_vocals_path = None
    for f in output_files:
        f_lower = f.lower()
        if 'no_' in f_lower or 'backing' in f_lower or 'instrumental' in f_lower or 'karaoke' in f_lower:
            backup_vocals_path = f
        else:
            main_vocals_path = f

    # Fallback
    if main_vocals_path is None and backup_vocals_path is None:
        # For karaoke models, first output is usually the "removed" stem (backup), second is main
        backup_vocals_path = output_files[0] if len(output_files) > 0 else None
        main_vocals_path = output_files[1] if len(output_files) > 1 else None
    elif main_vocals_path is None:
        main_vocals_path = [f for f in output_files if f != backup_vocals_path][0] if len(output_files) > 1 else None
    elif backup_vocals_path is None:
        backup_vocals_path = [f for f in output_files if f != main_vocals_path][0] if len(output_files) > 1 else None

    return main_vocals_path, backup_vocals_path


def apply_dereverb(vocals_path, output_dir, model_display_name=None, denoise=True):
    """
    Remove reverb/echo from vocals.

    Returns:
        dereverbed_vocals_path (or the original path if model is None/skipped)
    """
    if model_display_name is None:
        model_display_name = DEFAULT_DEREVERB_MODEL

    model_filename = _resolve_model_filename(model_display_name, DEREVERB_MODELS)
    if model_filename is None:
        # "None (Skip)" was selected
        return vocals_path

    output_files = separate_audio(vocals_path, output_dir, model_filename, denoise)

    # For de-reverb models: we want the "dry" (no reverb) output.
    # Typically the secondary stem is the cleaned/dry version.
    dereverbed_path = None
    for f in output_files:
        f_lower = f.lower()
        if 'no_' in f_lower or 'dry' in f_lower or 'dereverb' in f_lower:
            dereverbed_path = f

    # Fallback: for VR arch de-reverb models, the second output is usually the dry signal
    if dereverbed_path is None:
        dereverbed_path = output_files[1] if len(output_files) > 1 else output_files[0]

    return dereverbed_path


def separate_instrumental_stems(instrumental_path, output_dir, model_display_name=None):
    """
    Separate an instrumental track into individual instrument stems.

    Uses Demucs models via audio-separator to split the instrumental into
    drums, bass, guitar, piano, other, and vocal remnants.

    Args:
        instrumental_path: Path to the instrumental WAV file.
        output_dir: Directory to write output stem files.
        model_display_name: Display name from INSTRUMENTAL_STEM_MODELS catalog.

    Returns:
        dict: Mapping of stem name to file path, e.g.
              {"drums": "/path/drums.wav", "bass": ..., "vocal_remnants": ...}
    """
    if model_display_name is None:
        model_display_name = DEFAULT_STEM_MODEL
    model_filename = _resolve_model_filename(model_display_name, INSTRUMENTAL_STEM_MODELS)

    # Determine which stems this model produces
    is_6s = '6s' in model_filename if model_filename else False
    expected_stems = STEM_NAMES_6S if is_6s else STEM_NAMES_4S

    output_files = separate_audio(instrumental_path, output_dir, model_filename)

    # Map output files to stem names based on filename contents
    stem_paths = {}
    unmatched = []
    for f in output_files:
        f_lower = os.path.basename(f).lower()
        matched = False
        if 'drum' in f_lower:
            stem_paths['drums'] = f
            matched = True
        elif 'bass' in f_lower:
            stem_paths['bass'] = f
            matched = True
        elif 'guitar' in f_lower:
            stem_paths['guitar'] = f
            matched = True
        elif 'piano' in f_lower:
            stem_paths['piano'] = f
            matched = True
        elif 'vocal' in f_lower:
            # Vocal remnants from instrumental — save as labeled file
            stem_paths['vocal_remnants'] = f
            matched = True
        elif 'other' in f_lower or 'no_' not in f_lower:
            stem_paths['other'] = f
            matched = True

        if not matched:
            unmatched.append(f)

    # Assign any unmatched files to 'other' if not already set
    if 'other' not in stem_paths and unmatched:
        stem_paths['other'] = unmatched.pop(0)

    return stem_paths
