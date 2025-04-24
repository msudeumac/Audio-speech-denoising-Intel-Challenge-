import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm






def compute_mel(directory, max_files=None):
    """Compute Mel spectrogram for all .wav files in the directory"""
    mel_results = {}
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    for file in tqdm(wav_files[:max_files], desc="Processing audio files"):
        file_path = os.path.join(directory, file)
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Compute STFT
        stft_tensor = torchaudio.functional.spectrogram(
            waveform.squeeze(0),
            n_fft=1024,
            hop_length=512,
            window=torch.hann_window(1024),
            win_length=1024,
            power=1.0,
            normalized=False,
            pad=0

        )
        
        # Convert to mel scale
        mel_filter = torchaudio.transforms.MelScale(
            n_mels=64,  # Number of mel bands
            sample_rate=sample_rate,
            n_stft=stft_tensor.shape[0]
        )
        
        mel_spectrogram = mel_filter(torch.abs(stft_tensor))
        mel_results[file] = mel_spectrogram
        
    return mel_results


def plot_amplitude_distributions(clean_mels, noisy_mels, noise_mels):
    """
    Plot histograms of amplitude distributions for clean speech, noisy speech, and noise.
    
    Args:
        clean_mels (dict): Dictionary of clean mel spectrograms
        noisy_mels (dict): Dictionary of noisy mel spectrograms
        noise_mels (dict): Dictionary of noise mel spectrograms
    """
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Plot clean speech distribution
    plt.subplot(131)
    clean_values = np.concatenate([mel.numpy().flatten() for mel in clean_mels.values()])
    plt.hist(clean_values, bins=100, alpha=0.7, color='green', label='Clean Speech')
    plt.axvline(np.mean(clean_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(clean_values):.4f}')
    plt.title('Clean Speech Amplitude Distribution')
    plt.xlabel('Mel Spectrogram Values')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot noisy speech distribution
    plt.subplot(132)
    noisy_values = np.concatenate([mel.numpy().flatten() for mel in noisy_mels.values()])
    plt.hist(noisy_values, bins=100, alpha=0.7, color='red', label='Noisy Speech')
    plt.axvline(np.mean(noisy_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(noisy_values):.4f}')
    plt.title('Noisy Speech Amplitude Distribution')
    plt.xlabel('Mel Spectrogram Values')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot noise distribution
    plt.subplot(133)
    noise_values = np.concatenate([mel.numpy().flatten() for mel in noise_mels.values()])
    plt.hist(noise_values, bins=100, alpha=0.7, color='blue', label='Noise')
    plt.axvline(np.mean(noise_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(noise_values):.4f}')
    plt.title('Noise Amplitude Distribution')
    plt.xlabel('Mel Spectrogram Values')
    plt.ylabel('Count')
    plt.legend()
    
    # Add overall title and adjust layout
    plt.suptitle('Amplitude Distributions of Speech and Noise Signals', y=1.05)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('amplitude_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistical information
    print("\nStatistical Analysis:")
    print(f"Clean Speech - Mean: {np.mean(clean_values):.4f}, Std: {np.std(clean_values):.4f}")
    print(f"Noisy Speech - Mean: {np.mean(noisy_values):.4f}, Std: {np.std(noisy_values):.4f}")
    print(f"Noise - Mean: {np.mean(noise_values):.4f}, Std: {np.std(noise_values):.4f}")
    
    # Calculate and print percentiles
    percentiles = [10, 25, 35, 50, 75, 90]
    print("\nPercentile Analysis:")
    for p in percentiles:
        print(f"\n{p}th Percentile:")
        print(f"Clean Speech: {np.percentile(clean_values, p):.4f}")
        print(f"Noisy Speech: {np.percentile(noisy_values, p):.4f}")
        print(f"Noise: {np.percentile(noise_values, p):.4f}")

def main():
    # Load your mel spectrograms
    # Replace these paths with your actual paths
    speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
    noise_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_noise"
    output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"

    

     # Load and process audio files
    clean_mels = compute_mel(speech_dir)
    noise_mels = compute_mel(noise_dir)
    noisy_mels = compute_mel(output_dir)
    

    
    
    
    # Plot distributions
    plot_amplitude_distributions(clean_mels, noisy_mels, noise_mels)

if __name__ == "__main__":
    main()