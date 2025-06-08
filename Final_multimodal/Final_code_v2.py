def multimodal_audio_video_pipeline(path):
    import os
    import numpy as np
    import subprocess
    import soundfile as sf
    import copy
    from Audio.audio_extract import extract_audio_video_arrays
    from Audio.speech_detection import speech_detection
    from Audio.noise_detection import is_music
    from Audio.Final_audio_model import process_all_files
    from Video.deepfake_detection_v2 import run_video_authenticity_prediction
    from Video.face_detection_yolo8 import process_single_video
    
    
    base_dir = os.getcwd()
   

    def save_stereo_audio(audio_array, sample_rate, output_path):
        if audio_array.dtype == np.float64:
            audio_array = audio_array.astype(np.float32)

        if len(audio_array.shape) == 2 and audio_array.shape[1] == 2:
            sf.write(output_path, audio_array, sample_rate)
            print(f"Saved stereo WAV to: {output_path}")
        else:
            raise ValueError("Audio must be stereo (shape: [samples, 2])")



    def separate_vocals(audio_array, sample_rate):
        base_dir = os.getcwd()
        
        dirr = os.path.join(base_dir, "Audio", "vocal-remover")
        os.chdir(dirr)
        
        input_path = os.path.join(base_dir, "Audio", "vocal-remover", "separated_audio", "temp_audio.wav")

        if len(audio_array.shape) == 1:
            audio_array = np.expand_dims(audio_array, axis=1)

        sf.write(input_path, audio_array, sample_rate)
        
        path_denoised = os.path.join(base_dir, "Audio", "vocal-remover", "Audio_denoised")

        subprocess.run([
            "python", "inference.py",
            "--input", "separated_audio/temp_audio.wav",
            "--output_dir", path_denoised,
            "--gpu", "0"
        ])
        
        os.remove(input_path)

    # Step 1: Extract media
    media_type, audio_array, video_frames, sample_rate, frame_count = extract_audio_video_arrays(path)

    audio_result = "Audio not detected"
    audio_model_path = None

    if media_type in ["Audio and Video", "Audio"] and audio_array is not None and len(audio_array) > 0:
        try:
            separate_vocals(audio_array, sample_rate)
            
            noise_path = os.path.join(base_dir, "Audio", "vocal-remover", "Audio_denoised", "temp_audio_Instruments.mp3")
            
            vocals_path = os.path.join(base_dir, "Audio", "vocal-remover", "Audio_denoised", "temp_audio_Vocals.mp3")

            result_noise = is_music(noise_path)
            result_speech, speech_dur, total_dur = speech_detection(vocals_path)

            if result_speech:
                if result_noise == "music":
                    audio_model_path = vocals_path
                    print("Background music is detected")
                else:
#                    save_stereo_audio(audio_array, sample_rate, vocals_path)
                
#                    vocal_path=copy.copy(path)
                    audio_model_path = copy.copy(path)
                    print("There is no background music")
                    
                os.chdir(base_dir)
                P_totall, S_totall = process_all_files(audio_model_path)

                if S_totall[0][0] == 1:
                    print("Audio is fake")
                else:
                    print("Audio is real")

                audio_result = "Speech detected"
            else:
                print("Speech not detected")

        except Exception as e:
            print(f"Audio processing failed: {e}")

    else:
        print("No valid audio detected.")

    # --- VIDEO SECTION ---
    if media_type in ["Audio and Video", "Video"]:
        
        os.chdir(base_dir)
        path_detected = os.path.join(base_dir, "Video", "detected", "output.mp4")

        process_single_video(path, path_detected)
        run_video_authenticity_prediction(path_detected)
    else:
        print("Video not detected")
        
#    os.chdir(base_dir)
    return audio_result, media_type

path = r"/Users/mmli/Desktop/donut video2.mp4"
result = multimodal_audio_video_pipeline(path)
