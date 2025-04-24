import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
NUM_INPUT = 64  # Number of Mel bands

def compute_mel(directory, max_files=None):
    """Compute Mel spectrogram for all .wav files in the directory"""
    mel_results = {}
    wav_files = [file for file in os.listdir(directory) if file.endswith('.wav')]
    
    for file in tqdm(wav_files[:max_files], desc="Computing Mel Spectrograms"):
        file_path = os.path.join(directory, file)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Compute STFT
        stft_tensor = torchaudio.functional.spectrogram(
            waveform.squeeze(0),
            n_fft=1024,
            hop_length=512,
            pad=0,
            window=torch.hann_window(1024),
            win_length=1024,
            power=1.0,
            normalized=False
        )
        
        magnitude_spectrum = torch.abs(stft_tensor)
        
        mel_filter = torchaudio.transforms.MelScale(
            n_mels=NUM_INPUT,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=None,
            n_stft=magnitude_spectrum.shape[0]
        )
        
        mel_spectrogram = mel_filter(magnitude_spectrum)
        mel_results[file] = mel_spectrogram

    return mel_results




    
    





if __name__ == "__main__":
    # Directories
    speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
    noise_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_noise"
    output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"
    
    # Compute mel spectrograms
    print("Computing clean speech mel spectrograms...")
    clean_mels = compute_mel(speech_dir, max_files=None)
    
    print("\nComputing noise mel spectrograms...")
    noise_mels = compute_mel(noise_dir, max_files=None)
    
    print("\nComputing noisy speech mel spectrograms...")
    noisy_mels = compute_mel(output_dir, max_files=None)

    # Get values
    clean_mel = np.concatenate([mel.numpy().flatten() for mel in clean_mels.values()])
    noise_mel = np.concatenate([mel.numpy().flatten() for mel in noise_mels.values()])
    noisy_mel = np.concatenate([mel.numpy().flatten() for mel in noisy_mels.values()])


    print(f"\nNumber of files processed:")
    print(f"Clean files: {len(clean_mels)}")
    print(f"Noise files: {len(noise_mels)}")
    print(f"Noisy files: {len(noisy_mels)}")
    
    print(f"\nTotal values in distributions:")
    print(f"Clean values: {len(clean_mel)}")
    print(f"Noise values: {len(noise_mel)}")
    print(f"Noisy values: {len(noisy_mel)}")
    
    # Create distribution plot
    plt.figure(figsize=(15, 5))
    
    # Plot clean speech distribution
    plt.subplot(131)
    plt.hist(clean_mel, bins= 'fd', alpha=0.7, color='g', label='Clean')
    clean_mean = np.mean(clean_mel)
    clean_std = np.std(clean_mel)
    plt.axvline(clean_mean, color='r', linestyle='--', linewidth=2,
                label=f'Mean={clean_mean:.4f}')
    plt.axvline(clean_mean + clean_std, color='k', linestyle=':', linewidth=1,
                label=f'Mean±Std')
    plt.axvline(clean_mean - clean_std, color='k', linestyle=':', linewidth=1)
    plt.title('Clean Speech Distribution')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    # Plot noise distribution
    plt.subplot(132)
    plt.hist(noise_mel, bins= 'fd', alpha=0.7, color='r', label='Noise')
    noise_mean = np.mean(noise_mel)
    noise_std = np.std(noise_mel)
    plt.axvline(noise_mean, color='r', linestyle='--', linewidth=2,
                label=f'Mean={noise_mean:.4f}')
    plt.axvline(noise_mean + noise_std, color='k', linestyle=':', linewidth=1,
                label=f'Mean±Std')
    plt.axvline(noise_mean - noise_std, color='k', linestyle=':', linewidth=1)
    plt.title('Noise Distribution')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    # Plot noisy speech distribution
    plt.subplot(133)
    plt.hist(noisy_mel, bins= 'fd', alpha=0.7, color='b', label='Noisy')
    noisy_mean = np.mean(noisy_mel)
    noisy_std = np.std(noisy_mel)
    plt.axvline(noisy_mean, color='r', linestyle='--', linewidth=2,
                label=f'Mean={noisy_mean:.4f}')
    plt.axvline(noisy_mean + noisy_std, color='k', linestyle=':', linewidth=1,
                label=f'Mean±Std')
    plt.axvline(noisy_mean - noisy_std, color='k', linestyle=':', linewidth=1)
    plt.title('Noisy Speech Distribution')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mel_distributions_comparison.png')
    plt.show()

    # Print statistics for each
    print("\nClean Speech Statistics:")
    print(f"Mean: {np.mean(clean_mel):.4f}")
    print(f"Std: {np.std(clean_mel):.4f}")
    print(f"Min: {np.min(clean_mel):.4f}")
    print(f"Max: {np.max(clean_mel):.4f}")
    
    print("\nNoise Statistics:")
    print(f"Mean: {np.mean(noise_mel):.4f}")
    print(f"Std: {np.std(noise_mel):.4f}")
    print(f"Min: {np.min(noise_mel):.4f}")
    print(f"Max: {np.max(noise_mel):.4f}")
    
    print("\nNoisy Speech Statistics:")
    print(f"Mean: {np.mean(noisy_mel):.4f}")
    print(f"Std: {np.std(noisy_mel):.4f}")
    print(f"Min: {np.min(noisy_mel):.4f}")
    print(f"Max: {np.max(noisy_mel):.4f}")

   

