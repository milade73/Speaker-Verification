import os
import numpy as np
from moviepy import VideoFileClip, AudioFileClip

import warnings

def extract_audio_video_arrays(file_path):
    """
    Extracts audio and/or video from a file and returns them as separate arrays.
    
    Returns:
        Tuple: (media_type, audio_array, video_frames, sample_rate, frame_count)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File does not exist.")

    ext = os.path.splitext(file_path)[1].lower()
    audio_only_exts = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']

    audio_array = None
    video_frames = []
    sample_rate = None
    frame_count = None
    has_audio = False
    has_video = False

    # If audio-only extension, use AudioFileClip directly
    if ext in audio_only_exts:
        try:
            clip = AudioFileClip(file_path)
            audio_array = np.array(list(clip.iter_frames()))
            sample_rate = clip.fps
            has_audio = True
            clip.close()
        except Exception as e:
            warnings.warn(f"Failed to extract audio: {e}")
    else:
        try:
            clip = VideoFileClip(file_path)

            # Try extracting video frames
            try:
                video_frames = list(clip.iter_frames())
                frame_count = len(video_frames)
                has_video = True
            except Exception as e:
                warnings.warn(f"Video found but could not extract frames: {e}")

            # Try extracting audio
            if clip.audio:
                try:
                    audio_array = np.array(list(clip.audio.iter_frames()))
                    sample_rate = clip.audio.fps
                    has_audio = True
                except Exception as e:
                    warnings.warn(f"Audio found but could not extract: {e}")

            clip.close()

        except Exception as e:
            warnings.warn(f"Failed to load file with VideoFileClip: {e}")

    # Determine media type
    if has_audio and has_video:
        media_type = "Audio and Video"
    elif has_video and not has_audio:
        media_type = "Video"
    elif has_audio:
        media_type = "Audio"
    else:
        media_type = "Unsupported or Empty Media"

    return media_type, audio_array, video_frames, sample_rate, frame_count





















