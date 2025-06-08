from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import torchaudio
import numpy as np

def detect_speech(audio_array, sample_rate, threshold=0.3, target_sr=16000):
    """
    Detects whether speech in an audio array is significant.
    
    Args:
        audio_array (np.ndarray or torch.Tensor): 1D or 2D audio signal.
        sample_rate (int): Original sample rate.
        threshold (float): Ratio of speech duration to total duration.
        target_sr (int): Resample target (default 16000).
    
    Returns:
        bool: True if significant speech detected
        float: Speech duration (s)
        float: Total duration (s)
    """
    # Convert to torch tensor if needed
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array)
    else:
        audio_tensor = audio_array

    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor[0]  # use first channel

    audio_tensor = audio_tensor.float().cpu()

    # Resample to 16000 Hz if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)
        sample_rate = target_sr

    # Load model
    model = load_silero_vad()

    # Get timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=True
    )

    total_duration = audio_tensor.shape[0] / sample_rate
    speech_duration = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
    is_significant = (speech_duration / total_duration) >= threshold

    return is_significant, speech_duration, total_duration

import torchaudio


def speech_detection(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    result, speech_dur, total_dur = detect_speech(waveform, sr)

    print(f"\nTotal duration: {total_dur:.2f}s")
    print(f"Speech duration: {speech_dur:.2f}s")
    print("Speech detected." if result else "No significant speech detected.")
    return result, speech_dur, total_dur



