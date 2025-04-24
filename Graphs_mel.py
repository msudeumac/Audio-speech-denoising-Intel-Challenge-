# mel_spec.py

import os
import random
import torch
import torchaudio
import matplotlib.pyplot as plt

def compute_mel_from_file(file_path):
    """Compute Mel spectrogram for a specific file"""
    waveform, sample_rate = torchaudio.load(file_path)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=1024,  # Increased FFT size for better frequency resolution
        hop_length=512,
        n_mels=64,  # Number of Mel bands
        sample_rate= 16000,  # Use the actual sample rate
        f_min=0.0,
        f_max=None,  # Default is sample_rate / 2
        window_fn=torch.hann_window  # Pass the function, not a tensor
    )
    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram_db = 10 * torch.log10(mel_spectrogram + 1e-10)
    return mel_spectrogram_db[0, :, :].numpy()

if __name__ == "__main__":
    # Directories
    speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
    noise_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_noise"
    output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"
    
    # Verify directories exist
    for directory in [speech_dir, noise_dir, output_dir]:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            exit(1)
    
    # Get a random noisy file
    noisy_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    if not noisy_files:
        print(f"No .wav files found in {output_dir}")
        exit(1)
    noisy_file = random.choice(noisy_files)
    print(f"Selected noisy file: {noisy_file}")
    
    # Get corresponding clean file by removing last 12 characters and adding .wav
    clean_file = noisy_file[:-16] + '.wav'  # This will remove "_with_noise.wav" and add ".wav"
    clean_path = os.path.join(speech_dir, clean_file)
    print(f"Looking for clean file: {clean_path}")
    if not os.path.exists(clean_path):
        print(f"Clean file not found: {clean_path}")
        exit(1)
    
    # Get a random noise file
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    if not noise_files:
        print(f"No .wav files found in {noise_dir}")
        exit(1)
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)
    print(f"Selected noise file: {noise_path}")
    
    # Compute spectrograms
    clean_mel = compute_mel_from_file(os.path.join(speech_dir, clean_file))
    noise_mel = compute_mel_from_file(os.path.join(noise_dir, noise_file))
    noisy_mel = compute_mel_from_file(os.path.join(output_dir, noisy_file))
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=150)
    
    # Plot clean speech spectrogram
    im1 = axes[0].imshow(clean_mel, cmap='viridis', aspect='auto', origin='lower', interpolation='bilinear')
    axes[0].set_title(f'Clean Speech: {clean_file}')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
    
    # Plot noise spectrogram
    im2 = axes[1].imshow(noise_mel, cmap='viridis', aspect='auto', origin='lower', interpolation='bilinear')
    axes[1].set_title(f'Noise: {noise_file}')
    plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')
    
    # Plot noisy speech spectrogram
    im3 = axes[2].imshow(noisy_mel, cmap='viridis', aspect='auto', origin='lower', interpolation='bilinear')
    axes[2].set_title(f'Noisy Speech: {noisy_file}')
    plt.colorbar(im3, ax=axes[2], format='%+2.0f dB')
    
    # Add common labels
    fig.text(0.04, 0.5, 'Mel Frequency Bins', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Time (frames)', ha='center')
    
    plt.tight_layout()
    plt.savefig('MEL_spectrograms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

