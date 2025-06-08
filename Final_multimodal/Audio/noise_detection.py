import librosa
import numpy as np

def is_music(audio_path, threshold_flatness=0.3, threshold_zcr=0.1, threshold_energy=0.002):
    y, sr = librosa.load(audio_path, sr=None)

    # Feature 1: Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Feature 2: Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Feature 3: Energy (RMS)
    energy = np.mean(librosa.feature.rms(y=y))

    # Print values and threshold checks
#    print(f"Spectral flatness: {flatness:.3f} (threshold: {threshold_flatness}) → {'PASS' if flatness < threshold_flatness else 'FAIL'}")
#    print(f"Zero Crossing Rate: {zcr:.3f} (threshold: {threshold_zcr}) → {'PASS' if zcr < threshold_zcr else 'FAIL'}")
#    print(f"Energy: {energy:.5f} (threshold: {threshold_energy}) → {'PASS' if energy > threshold_energy else 'FAIL'}")

    # Apply soft rule: at least 2 out of 3 must pass
    pass_count = sum([
        flatness < threshold_flatness,
        zcr < threshold_zcr,
        energy > threshold_energy
    ])

    if pass_count >= 2:
        return "music"
    else:
        return "noise"

# Example usage
#audio_path = r"E:\NLP\Final_multi_modal\Audio\vocal-remover\Audio_denoised\temp_audio_Instruments.mp3"
#result = is_music(audio_path)
#print("Detected as:", result)
