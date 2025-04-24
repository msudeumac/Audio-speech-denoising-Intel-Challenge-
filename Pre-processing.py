import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import time
import random
import torchaudio
import fnmatch
import torch 

def match_lengths(speech_waveform, noise_waveform):
    """Ensure both waveforms have the same length by truncating the longer one"""
    speech_len = speech_waveform.size(1)
    noise_len = noise_waveform.size(1)
    
    # Truncate to the shortest length
    min_len = min(speech_len, noise_len)
    speech_waveform = speech_waveform[:, :min_len]
    noise_waveform = noise_waveform[:, :min_len]
    
    return speech_waveform, noise_waveform

def check_space(directory, required_gb):
    """Check if enough space is available"""
    _, _, free = shutil.disk_usage(directory)
    free_gb = free / (1024**3)
    return free_gb >= required_gb

def resample_audio(waveform, original_sample_rate, target_sample_rate):
    """Resample audio to a target sample rate."""
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def process_in_batches(speech_dir, noise_dir, output_dir, snr_db=0, preprocessed_speech_dir=None, preprocessed_noise_dir=None):
    """Process files in smaller batches with detailed progress tracking"""
    
    target_sample_rate = 16000  # Set target sample rate

    # Create directories for pre-processed files if they don't exist
    os.makedirs(preprocessed_speech_dir, exist_ok=True)
    os.makedirs(preprocessed_noise_dir, exist_ok=True)

    # Get file lists
    speech_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

    # First load and resample all files to get their lengths
    print("Loading and resampling files...")
    speech_waveforms = []
    for speech_file in tqdm(speech_files, desc="Loading speech files"):
        speech_path = os.path.join(speech_dir, speech_file)
        waveform, sample_rate = torchaudio.load(speech_path)
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        speech_waveforms.append((speech_file, waveform))

    noise_waveforms = []
    for noise_file in tqdm(noise_files, desc="Loading noise files"):
        noise_path = os.path.join(noise_dir, noise_file)
        waveform, sample_rate = torchaudio.load(noise_path)
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        noise_waveforms.append((noise_file, waveform))

    # Find minimum length across all files
    min_length = float('inf')
    for _, waveform in speech_waveforms + noise_waveforms:
        min_length = min(min_length, waveform.size(1))

    # Save truncated speech files
    print("Saving truncated speech files...")
    for speech_file, speech_waveform in tqdm(speech_waveforms, desc="Processing speech files"):
        # Truncate to minimum length
        truncated_speech = speech_waveform[:, :min_length]
        
        # Save truncated speech file
        speech_output_name = f"{speech_file}"
        torchaudio.save(
            os.path.join(preprocessed_speech_dir, speech_output_name),
            truncated_speech,
            target_sample_rate
        )

    # Save truncated noise files
    print("Saving truncated noise files...")
    for noise_file, noise_waveform in tqdm(noise_waveforms, desc="Processing noise files"):
        # Truncate to minimum length
        truncated_noise = noise_waveform[:, :min_length]
        
        # Save truncated noise file
        noise_output_name = f"{noise_file}"
        torchaudio.save(
            os.path.join(preprocessed_noise_dir, noise_output_name),
            truncated_noise,
            target_sample_rate
        )

    print("Processing complete. Resampled and truncated files saved to preprocessed directories.")



def combine_batch(speech_files, noise_files, preprocessed_speech_dir, preprocessed_noise_dir, output_dir, snr_db, max_noise_files, target_sample_rate=16000):
    """
    Process one batch of combinations with progress tracking.
    Args:
        speech_files (list): List of speech file names to be processed.
        noise_files (list): List of noise file names to be processed.
        preprocessed_speech_dir (str): Directory containing preprocessed speech files.
        preprocessed_noise_dir (str): Directory containing preprocessed noise files.
        output_dir (str): Directory where the combined output files will be saved.
        snr_db (float): Signal-to-noise ratio in decibels for combining speech and noise.
        max_noise_files (int): Maximum number of noise files to combine with each speech file.
        target_sample_rate (int, optional): Target sample rate for audio files. Defaults to 16000.
    Returns:
        int: Number of files successfully processed and combined.
    """
    """Process one batch of combinations with progress tracking"""
    
    total = len(speech_files) * max_noise_files  # Adjust total based on max noise files
    files_processed = 0
    
    with tqdm(total=total, desc="Combining files") as pbar:
        for speech_file in speech_files:
            try:
                # Construct the base name for the speech file
                base_speech_name = speech_file
                
                # Find the actual file in the speech directory
                matching_files = fnmatch.filter(os.listdir(preprocessed_speech_dir), f"{base_speech_name}*")
                
                if not matching_files:
                    print(f"Warning: No matching speech file for {base_speech_name} in {preprocessed_speech_dir}.")
                    pbar.update(max_noise_files)  # Update progress for skipped noise files
                    continue
                
                # Load the first matching speech file
                speech_path = os.path.join(preprocessed_speech_dir, matching_files[0])
                speech_waveform, speech_sr = torchaudio.load(speech_path)
                
                # Resample if necessary
                if speech_sr != target_sample_rate:
                    speech_waveform = resample_audio(speech_waveform, speech_sr, target_sample_rate)
                
               
                # Convert speech to numpy and normalize - do this BEFORE the noise loop
                speech_waveform_np = speech_waveform.numpy()
                speech_waveform_np = normalize_for_snn(speech_waveform_np)
                
                
                for noise_file in noise_files[:max_noise_files]:  # Limit to max_noise_files
                    try:

                        # Construct the base name pattern for noise files
                        base_noise_name_pattern = noise_file
                        
                        # Find matching noise files in the noise directory
                        matching_noise_files = fnmatch.filter(os.listdir(preprocessed_noise_dir), base_noise_name_pattern)
                        
                        if not matching_noise_files:
                            print(f"Warning: No matching noise files for {base_noise_name_pattern} in {preprocessed_noise_dir}.")
                            pbar.update(max_noise_files)  # Update progress for skipped noise files
                            continue
                        # Load the noise file
                        noise_path = os.path.join(preprocessed_noise_dir, matching_noise_files[0])
                        noise_waveform, noise_sr = torchaudio.load(noise_path)
                        
                        # Resample if necessary
                        if noise_sr != target_sample_rate:
                            noise_waveform = resample_audio(noise_waveform, noise_sr, target_sample_rate)
                        
                        # Convert to numpy and normalize
                        noise_waveform_np = noise_waveform.numpy()
                        noise_waveform_np = normalize_for_snn(noise_waveform_np)
                        
                        # Match lengths
                        min_len = min(speech_waveform_np.shape[1], noise_waveform_np.shape[1])
                        speech_waveform_np = speech_waveform_np[:, :min_len]
                        noise_waveform_np = noise_waveform_np[:, :min_len]

                        # Combine signals
                        combined = combine_signals(speech_waveform_np, noise_waveform_np, snr_db)
                        

                        # Ensure combined is a torch tensor
                        if not isinstance(combined, torch.Tensor):
                            combined = torch.from_numpy(combined).float()
                        
                        # Save the combined audio
                        output_name = f"{Path(speech_file).stem}_{Path(noise_file).stem}.wav"
                        output_path = os.path.join(output_dir, output_name)
                        torchaudio.save(output_path, combined, target_sample_rate)

                        files_processed += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError processing noise file {noise_file}: {e}")
                        pbar.update(1)

            except Exception as e:
                print(f"\nError processing speech file {speech_file}: {e}")
                pbar.update(1)
    
    
                    
            # except Exception as e:
            #     print(f"\nError processing speech file {speech_file}: {e}")
            #     if 'matching_noise_files' in locals():
            #         pbar.update(len(matching_noise_files))  # Update progress for all selected noise files
            #     else:
            #         pbar.update(max_noise_files)  # Fallback to max_noise_files if not defined
    
    return files_processed

def normalize_for_snn(audio_data):
    """Normalize audio for SNN training"""
    audio_data = audio_data - np.mean(audio_data)
    std = np.std(audio_data)
    if std < 1e-10:
        return np.zeros_like(audio_data)
    audio_data = audio_data / (3 * std)
    audio_data = np.tanh(audio_data)
    return audio_data

# ... (previous code remains the same)

def combine_signals(speech, noise, snr_db):
    """Combine speech and noise at specified SNR with normalization"""
    # Convert inputs to numpy arrays if they aren't already
    speech = np.asarray(speech)
    noise = np.asarray(noise)
    
    # Calculate signal powers
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # Calculate scaling factor for noise
    scaling = np.sqrt((speech_power / (noise_power + epsilon)) * 10 ** (-snr_db / 10))
    
    # Combine signals
    noisy_speech = speech + scaling * noise
    
    # Normalize the combined signal
    noisy_speech = normalize_for_snn(noisy_speech)
    
    # Convert back to torch tensor
    noisy_speech = torch.from_numpy(noisy_speech).float()
    
    return noisy_speech



if __name__ == "__main__":
    # Configuration
    speech_dir = "/its/home/msu22/attempt_DISS/extracted_clean_emotional_speech_files/datasets_fullband/clean_fullband/emotional_speech/crema_d"
    noise_dir = "/its/home/msu22/attempt_DISS/extracted_noise_files/datasets_fullband/noise_fullband"
    output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"
    preprocessed_speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
    preprocessed_noise_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_noise"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_speech_dir, exist_ok=True)
    os.makedirs(preprocessed_noise_dir, exist_ok=True)
    
    # Process files with a limited number of noise files
    process_in_batches(speech_dir, noise_dir, output_dir, 
                      snr_db=0,      # Adjust SNR as needed
                    #   batch_size=5, # Adjust batch size based on your memory
                      preprocessed_speech_dir=preprocessed_speech_dir,
                      preprocessed_noise_dir=preprocessed_noise_dir)
    
    
    max_noise_files = 5  # Define how many noise files to use per speech file
    combine_batch(
        speech_files=os.listdir(preprocessed_speech_dir),
        noise_files=os.listdir(preprocessed_noise_dir),
        preprocessed_speech_dir=preprocessed_speech_dir,
        preprocessed_noise_dir=preprocessed_noise_dir,
        output_dir=output_dir,
        snr_db=0,  # Signal-to-noise ratio
        max_noise_files=max_noise_files,
    )

