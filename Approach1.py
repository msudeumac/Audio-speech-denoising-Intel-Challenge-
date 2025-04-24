import os
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)
import torchaudio
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import (OptimiserParamSchedule, SpikeRecorder, VarRecorder)
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.utils.data import preprocess_spikes
from ml_genn.compilers.event_prop_compiler import default_params
from time import perf_counter
from tqdm import tqdm 
from ml_genn.synapses import Exponential
from matplotlib.gridspec import GridSpec


# Constants
NUM_INPUT = 64  # Number of Mel bands 
NUM_HIDDEN = 128
NUM_OUTPUT = 64  # Should match NUM_INPUT since we're trying to reconstruct clean spectrograms
NUM_EPOCHS = 50
TIME_STEPS = 45  # Match with example_timesteps - IT IS THE NUM OF TIME FRAMES 







def calculate_speech_quality_metrics(denoised, clean):
    """Calculate multiple metrics for speech quality assessment with error handling"""
    metrics = {}
    
    # 1. Signal-to-Noise Ratio (SNR)
    def calculate_snr(denoised, clean):
        signal_power = np.mean(clean ** 2)
        noise = denoised - clean
        noise_power = np.mean(noise ** 2)
        
        # Avoid division by zero
        if noise_power < 1e-10:  # Small threshold
            return 0.0
        return 10 * np.log10(signal_power / noise_power)
    
    # 2. Peak Signal-to-Noise Ratio (PSNR)
    def calculate_psnr(denoised, clean):
        mse = np.mean((denoised - clean) ** 2)
        if mse < 1e-10:  # Avoid division by zero
            return 100.0  # Return high value for perfect match
        max_signal = np.max(clean)
        if max_signal == 0:  # Avoid log of zero
            return 0.0
        return 20 * np.log10(max_signal / np.sqrt(mse))
    
    # 3. Mean Square Error (MSE)
    def calculate_mse(denoised, clean):
        return np.mean((denoised - clean) ** 2)
    
    # 4. Segmental SNR (SSNR)
    def calculate_segmental_snr(denoised, clean, segment_size=45):
        num_segments = len(clean) // segment_size
        ssnr_values = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            segment_clean = clean[start:end]
            segment_denoised = denoised[start:end]
            
            signal_power = np.mean(segment_clean ** 2)
            noise_power = np.mean((segment_denoised - segment_clean) ** 2)
            
            # Only append valid values
            if noise_power > 1e-10 and signal_power > 1e-10:
                ssnr = 10 * np.log10(signal_power / noise_power)
                ssnr_values.append(ssnr)
        
        # Return average only if we have values
        if len(ssnr_values) > 0:
            return np.mean(ssnr_values)
        return 0.0  # Default value if no valid segments
    
    # Calculate all metrics
    metrics['SNR'] = calculate_snr(denoised, clean)
    metrics['PSNR'] = calculate_psnr(denoised, clean)
    metrics['MSE'] = calculate_mse(denoised, clean)
    metrics['SSNR'] = calculate_segmental_snr(denoised, clean)
    
    return metrics

def normalize_mel_spectrogram(mel_spectrogram):
    """Normalize mel spectrogram to [0,1] range"""
    min_val = np.min(mel_spectrogram)
    max_val = np.max(mel_spectrogram)
    return (mel_spectrogram - min_val) / (max_val - min_val + 1e-10)

def compute_mel(directory, max_files=5060):
    """Compute Mel spectrogram for all .wav files in the given directory using PyTorch"""
    mel_results = {}
    
    # Get the list of .wav files in the directory
    wav_files = [file for file in os.listdir(directory) if file.endswith('.wav')]
    
    # Wrap the file processing loop with tqdm, limiting the total to max_files if provided
    for idx, file in enumerate(tqdm(wav_files[:max_files], desc="Computing Mel Spectrograms")):
        file_path = os.path.join(directory, file)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Compute STFT
        stft_tensor = torchaudio.functional.spectrogram(
            waveform.squeeze(0),  # Remove channel dimension
            n_fft=1024,
            hop_length=512,
            pad=0,
            window=torch.hann_window(1024),  # Use Hann window
            win_length=1024,
            power=1.0,
            normalized=False
        )
        
        magnitude_spectrum = torch.abs(stft_tensor)
        
        # Create a Mel filter bank
        mel_filter = torchaudio.transforms.MelScale(
            n_mels=NUM_INPUT,  # Number of Mel bands
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=None,  # Default is sample_rate / 2
            n_stft=magnitude_spectrum.shape[0]  # Number of STFT bins
        )
        
        # Apply the Mel filter bank to the magnitude spectrum
        mel_spectrogram = mel_filter(magnitude_spectrum)
        
        # Normalize the mel spectrogram
        mel_spectrogram_np = mel_spectrogram.numpy()
        normalized_mel = normalize_mel_spectrogram(mel_spectrogram_np)
        mel_results[file] = torch.from_numpy(normalized_mel)

    return mel_results

def analyze_mel_spectrograms(noisy_dir, clean_dir, noise_dir, max_files=None):
    """
    Analyze mel spectrograms from both directories
    Args:
        noisy_dir: Directory with noisy audio files
        clean_dir: Directory with clean audio files
        max_files: Number of files to analyze
    """
    # Compute mel spectrograms
    print("Computing clean speech mel spectrograms...")
    clean_mels = compute_mel(clean_dir, max_files=max_files)
    
    print("\nComputing noisy speech mel spectrograms...")
    noisy_mels = compute_mel(noisy_dir, max_files=max_files)

    print("Computing noise mel spectrograms...")
    noise_mels = compute_mel(noise_dir, max_files=max_files)
    
    # Analyze values
    all_noisy_values = []
    all_clean_values = []
    all_noise_values = []

    for file in clean_mels.keys():
        clean_mel = clean_mels[file].numpy()
        clean_mean_val= np.mean(clean_mel)
        all_clean_values.append(clean_mean_val)
        print(f"File: {file}")
        print(f"Mean: {clean_mean_val:.4f}")
        print(f"Shape: {clean_mel.shape}")
        print(f"Min: {np.min(clean_mel):.4f}")
        print(f"Max: {np.max(clean_mel):.4f}")



    
    for file in noisy_mels.keys():
        noisy_mel = noisy_mels[file].numpy()
        noisy_mean_val= np.mean(noisy_mel)
        all_noisy_values.append(noisy_mean_val)
        print(f"File: {file}")
        print(f"Mean: {noisy_mean_val:.4f}")
        print(f"Shape: {noisy_mel.shape}")
        print(f"Min: {np.min(noisy_mel):.4f}")
        print(f"Max: {np.max(noisy_mel):.4f}")


    for file in noise_mels.keys():
        noise_mel = noise_mels[file].numpy()
        noise_mean_val= np.mean(noise_mel)
        all_noise_values.append(noise_mean_val)
        print(f"File: {file}")
        print(f"Mean: {noise_mean_val:.4f}")
        print(f"Shape: {noise_mel.shape}")
        print(f"Min: {np.min(noise_mel):.4f}")
        print(f"Max: {np.max(noise_mel):.4f}")


    print("\nOverall Statistics of Clean Mel:")
    print(f"Average mean across files: {np.mean(all_clean_values):.4f}")
    print(f"Std of means: {np.std(all_clean_values):.4f}")
    print(f"Min mean: {np.min(all_clean_values):.4f}")
    print(f"Max mean: {np.max(all_clean_values):.4f}")

    print("\nOverall Statistics of Noisy Mel:")
    print(f"Average mean across files: {np.mean(all_noisy_values):.4f}")
    print(f"Std of means: {np.std(all_noisy_values):.4f}")
    print(f"Min mean: {np.min(all_noisy_values):.4f}")
    print(f"Max mean: {np.max(all_noisy_values):.4f}")

    print("\nOverall Statistics of Noise Mel:")
    print(f"Average mean across files: {np.mean(all_noise_values):.4f}")
    print(f"Std of means: {np.std(all_noise_values):.4f}")
    print(f"Min mean: {np.min(all_noise_values):.4f}")
    print(f"Max mean: {np.max(all_noise_values):.4f}")
    
    

    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    
    plt.subplot(131)
    plt.hist(all_clean_values, bins=50, alpha=0.7, label='Clean')
    plt.axvline(np.mean(all_clean_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(all_clean_values):.4f}')
    plt.axvline(np.mean(all_clean_values) + np.std(all_clean_values), color='k', linestyle=':', linewidth=1)
    plt.axvline(np.mean(all_clean_values) - np.std(all_clean_values), color='k', linestyle=':', linewidth=1)
    plt.title('All Clean Mel Distributions')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.legend()

    # plot the distribution of the noise mel 
    plt.subplot(132)
    plt.hist(all_noise_values, bins=50, alpha=0.7, label='Noise')
    plt.axvline(np.mean(all_noise_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(all_noise_values):.4f}')
    plt.axvline(np.mean(all_noise_values) + np.std(all_noise_values), color='k', linestyle=':', linewidth=1)
    plt.axvline(np.mean(all_noise_values) - np.std(all_noise_values), color='k', linestyle=':', linewidth=1)
    plt.title('All Noise Mel Distributions')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(133)
    plt.hist(all_noisy_values, bins=50, alpha=0.7, label='Noisy')
    plt.axvline(np.mean(all_noisy_values), color='r', linestyle='--', 
                label=f'Mean={np.mean(all_noisy_values):.4f}')
    plt.axvline(np.mean(all_noisy_values) + np.std(all_noisy_values), color='k', linestyle=':', linewidth=1)
    plt.axvline(np.mean(all_noisy_values) - np.std(all_noisy_values), color='k', linestyle=':', linewidth=1)
    plt.title('All Noisy Mel Distributions')
    plt.xlabel('Mel Values')
    plt.ylabel('Count')
    plt.legend()

    
    plt.tight_layout()
    plt.savefig('mel_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def apply_percentile_threshold(mel_spectrogram, percentile=90): 
    """Apply percentile-based thresholding to mel spectrogram change the percentile to 40 for more aggressive thresholding"""
    threshold = np.percentile(mel_spectrogram, percentile)
    return np.where(mel_spectrogram > threshold, mel_spectrogram, 0)

def visualize_thresholds(noisy_mel, clean_mel, percentiles=[20, 30, 35, 80, 85, 90, 95]):
    """Visualize different percentile thresholds"""
    num_thresholds = len(percentiles)
    fig, axes = plt.subplots(num_thresholds + 2, 1, figsize=(12, 4*(num_thresholds + 2)))
    
    # Plot original clean spectrogram
    im0 = axes[0].imshow(clean_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Clean Speech')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot original noisy spectrogram
    im1 = axes[1].imshow(noisy_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Noisy Speech (Original)')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot different thresholds
    for idx, p in enumerate(percentiles):
        thresholded = apply_percentile_threshold(noisy_mel, percentile=p)
        im = axes[idx+2].imshow(thresholded, aspect='auto', origin='lower', cmap='viridis')
        axes[idx+2].set_title(f'Threshold at {p}th percentile')
        plt.colorbar(im, ax=axes[idx+2])
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def filter_spikes(mel_spectrogram, percentile=90):
    """
    Filter spikes based on:
    1. Percentile threshold of mel spectrogram values
    2. Return only significant spikes
    """
    # First apply percentile threshold to mel spectrogram
    threshold = np.percentile(mel_spectrogram, percentile)
    thresholded_mel = np.where(mel_spectrogram > threshold, mel_spectrogram, 0)
    
    # Get indices of non-zero values (these will generate spikes)
    spike_indices = np.nonzero(thresholded_mel)
    spike_values = thresholded_mel[spike_indices]
    
    # Create spike times and IDs only for significant values
    times = spike_values.flatten()

    # # Create IDs in the same way as noisy/clean plots (0-63 range)
    # ids = np.arange(thresholded_mel.shape[1])  # Creates IDs from 0 to 63
    # ids = np.tile(ids, thresholded_mel.shape[0])  # Repeats the pattern

    #  # Create IDs using np.repeat (0-63 range)
    ids = spike_indices[1]  % 64
    
    # Preprocess these filtered spikes
    return preprocess_spikes(times, ids, thresholded_mel.shape[1])

def visualize_spike_filtering(noisy_mel, clean_mel, percentile=35):
    """Visualize spike filtering effect on noisy vs clean signals"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Original noisy spikes
    flattened_noisy = noisy_mel.reshape(-1)
    orig_ids = np.repeat(np.arange(noisy_mel.shape[1]), noisy_mel.shape[0])
    noisy_spikes = preprocess_spikes(flattened_noisy, orig_ids, noisy_mel.shape[1])
    
    # Filtered noisy spikes
    filtered_spikes = filter_spikes(noisy_mel, percentile)
    
    # Clean spikes for comparison
    flattened_clean = clean_mel.reshape(-1)
    clean_spikes = preprocess_spikes(flattened_clean, orig_ids, clean_mel.shape[1])
    
    # Plot original noisy spikes
    axes[0].scatter(noisy_spikes.spike_times, orig_ids[:len(noisy_spikes.spike_times)], 
                   alpha=0.5, s=1, c='red', label='Noisy')
    axes[0].set_title('Original Noisy Spikes')
    axes[0].set_ylabel('Neuron ID')
    
    # Plot filtered spikes
    axes[1].scatter(filtered_spikes.spike_times, 
                   np.arange(len(filtered_spikes.spike_times)), 
                   alpha=0.5, s=1, c='blue', label='Filtered')
    axes[1].set_title(f'Filtered Spikes (percentile={percentile})')
    axes[1].set_ylabel('Neuron ID')
    
    # Plot clean spikes
    axes[2].scatter(clean_spikes.spike_times, orig_ids[:len(clean_spikes.spike_times)], 
                   alpha=0.5, s=1, c='green', label='Clean')
    axes[2].set_title('Clean Speech Spikes')
    axes[2].set_ylabel('Neuron ID')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('spike_filtering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()






# Add before line 229 (before the epoch loop)
def validate_network(network, val_inputs, val_outputs, batch_size=16):
    """Run validation on the entire validation set"""
    total_val_loss = 0
    num_batches = 0
    
    for i in range(0, len(val_inputs), batch_size):
        batch_end = min(i + batch_size, len(val_inputs))
        
        # Process validation batch
        val_batch_inputs = val_inputs[i:batch_end]
        flattened_val_inputs = val_batch_inputs.reshape(-1)
        val_input_ids = np.repeat(np.arange(NUM_INPUT), TIME_STEPS * len(val_batch_inputs))
        val_batch_spikes = preprocess_spikes(flattened_val_inputs, val_input_ids, NUM_INPUT)
        
        val_batch_outputs = val_outputs[i:i+1]
        
        # Get network predictions (without training)
        metrics, cb_data = network.train(
            {input_population: [val_batch_spikes]},
            {output_population: val_batch_outputs},
            num_epochs=1,
            callbacks=callbacks
        )
        
        # Calculate validation loss
        val_output = cb_data["output_v"][-1]
        val_error = val_output - val_batch_outputs.reshape(-1, NUM_OUTPUT)
        batch_loss = np.mean(val_error * val_error)
        
        total_val_loss += batch_loss
        num_batches += 1
    
    return total_val_loss / num_batches

def evaluate_on_test_set(network, test_inputs, test_outputs, batch_size=16):
    """Evaluate model on test set and return metrics"""
    test_losses = []
    denoised_outputs = []
    
    for i in range(0, len(test_inputs), batch_size):
        batch_end = min(i + batch_size, len(test_inputs))
        
        # Process test batch
        test_batch_inputs = test_inputs[i:batch_end]
        flattened_test_inputs = test_batch_inputs.reshape(-1)
        test_input_ids = np.repeat(np.arange(NUM_INPUT), TIME_STEPS * len(test_batch_inputs))
        test_batch_spikes = preprocess_spikes(flattened_test_inputs, test_input_ids, NUM_INPUT)
        
        test_batch_outputs = test_outputs[i:i+1]
        
        # Get predictions
        metrics, cb_data = network.train(
            {input_population: [test_batch_spikes]},
            {output_population: test_batch_outputs},
            num_epochs=1,
            callbacks=callbacks
        )
        
        # Store outputs for audio comparison
        denoised_output = cb_data["output_v"][-1]
        denoised_outputs.append(denoised_output)
        
        # Calculate test loss
        error = denoised_output - test_batch_outputs.reshape(-1, NUM_OUTPUT)
        test_loss = np.mean(error ** 2)
        test_losses.append(test_loss)
    
    return np.mean(test_losses), denoised_outputs





if __name__ == "__main__":
    # Directories
    speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
    noise_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_noise"
    output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"
    
    # Set the maximum number of files to process
    max_files_to_process = 5060  # Change this value as needed- to match the lengths of input and output files (no of  )
    
    # Compute normalized mel spectrograms
    comb_mel_results = compute_mel(output_dir, max_files=max_files_to_process)
    clean_mel_results = compute_mel(speech_dir, max_files=max_files_to_process)

    # Prepare inputs and outputs
    comb_inputs = []
    clean_outputs = []

    for file in clean_mel_results.keys():
        # Clean outputs are already normalized from compute_mel

        thresholded_input_clean = apply_percentile_threshold(clean_mel_results[file].numpy(), percentile=90)
        clean_outputs.append(thresholded_input_clean)

    for file in comb_mel_results.keys():
        # Apply thresholding to normalized input
        thresholded_input = apply_percentile_threshold(comb_mel_results[file].numpy(), percentile=90)
        comb_inputs.append(thresholded_input)

    # Convert to numpy arrays
    comb_inputs = np.array(comb_inputs)
    clean_outputs = np.array(clean_outputs)

     # Now you can use comb_inputs and clean_outputs for further processing or training
    print(f"Original input shape: {comb_inputs.shape}, Output shape: {clean_outputs.shape}")


    
 




    inputs = comb_inputs.transpose(0, 2, 1)  # From (1000, 64, 45) to (1000, 45, 64)
    outputs = clean_outputs.transpose(0, 2, 1)  # From (1000, 64, 45) to (1000, 45, 64)
    
    print(f"Reshaped input shape: {inputs.shape}, Output shape: {outputs.shape}")


        # After line 42 (after the transpose operations)
    from sklearn.model_selection import train_test_split

    # First split: separate test set (80% train+val, 20% test)
    train_val_inputs, test_inputs, train_val_outputs, test_outputs = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )

    # Second split: separate train and validation
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        train_val_inputs, train_val_outputs, test_size=0.1, random_state=42
    )

    print("\nData split sizes:")
    print(f"Training samples: {len(train_inputs)}")      # ~70% of total
    print(f"Validation samples: {len(val_inputs)}")      # ~10% of total
    print(f"Test samples: {len(test_inputs)}")          # ~20% of total
            
 

    TAU_SYN = 5.016


     # Use it before training
    # sample_noisy = inputs[0].reshape(TIME_STEPS, NUM_INPUT)
    # sample_clean = outputs[0].reshape(TIME_STEPS, NUM_INPUT)
    analyze_mel_spectrograms(output_dir, speech_dir, noise_dir)
    
    sample_noisy = comb_inputs[0].reshape(TIME_STEPS, NUM_INPUT)
    sample_clean = clean_outputs[0].reshape(TIME_STEPS, NUM_INPUT)
    visualize_thresholds(sample_noisy, sample_clean)

    sample_noisy = comb_inputs[0].reshape(TIME_STEPS, NUM_INPUT)
    sample_clean = clean_outputs[0].reshape(TIME_STEPS, NUM_INPUT)
    
    
    visualize_spike_filtering(sample_noisy, sample_clean)


    MAX_SPIKES= TIME_STEPS * NUM_INPUT * 5060
    # Prepare the network
    network = Network(default_params)
    with network:
        # Populations
        input_population = Population(SpikeInput(max_spikes=MAX_SPIKES), NUM_INPUT, record_spikes=True)
        hidden_population = Population(LeakyIntegrateFire(v_thresh= 1.05138668, tau_mem=24.988, tau_refrac= None), NUM_HIDDEN, record_spikes=True)
        output_population = Population(LeakyIntegrate(tau_mem=24.988, readout="var"), NUM_OUTPUT, record_spikes=True)
        
        # Connections
        Connection(input_population, hidden_population, Dense(Normal(sd=2.5 / np.sqrt(NUM_INPUT))), Exponential(TAU_SYN))
        Connection(hidden_population, hidden_population, Dense(Normal(sd=1.5 / np.sqrt(NUM_HIDDEN))), Exponential(TAU_SYN))
        Connection(hidden_population, output_population, Dense(Normal(sd=2.0 / np.sqrt(NUM_HIDDEN))), Exponential(TAU_SYN))

    # Compile the network
    compiler = EventPropCompiler(
        example_timesteps=TIME_STEPS, 
        losses="mean_square_error", 
        optimiser=Adam(0.004041),  # Reduced learning rate
        reg_lambda_upper=1e-8, 
        reg_lambda_lower=1e-8,
        reg_nu_upper=10, 
        max_spikes=MAX_SPIKES
    )
    compiled_net = compiler.compile(network)



    with compiled_net:
        def alpha_schedule(epoch, alpha):
            if (epoch % 2) == 0 and epoch != 0:
                return alpha * 0.7
            else:
                return alpha
            
        

        # Define callbacks without progress bar
        callbacks = [
            VarRecorder(output_population, "v", key="output_v"),
            SpikeRecorder(input_population, key="input_spikes"),
            SpikeRecorder(hidden_population, key="hidden_spikes"),
            OptimiserParamSchedule("alpha", alpha_schedule)
        ]  # Removed "batch_progress_bar"

        
        
        # # Initialize storage for MSE and best model performance
        # mse_per_epoch = []
        # best_mse = float('inf')
        # patience = 10  # Number of epochs to wait for improvement
        # patience_counter = 0

        

        
        # Create better visualization setup using seaborn
        import seaborn as sns
        sns.set_theme(style="darkgrid")
        sns.set_context("notebook", font_scale=1.2)
        
        # Create figure for meaningful visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: MSE Learning Curve
        ax_mse = axes[0, 0]
        ax_mse.set_title('Training Progress')
        ax_mse.set_xlabel('Epoch')
        ax_mse.set_ylabel('MSE')
        
        
        
        # Plot 3: Network Activity
        ax_spikes = axes[1, 0]
        ax_spikes.set_title('Neural Activity')
        ax_spikes.set_xlabel('Time')
        ax_spikes.set_ylabel('Neuron Index')


        # Plot 2: Spectrogram Comparison
        ax_spectro = axes[0, 1]
        ax_spectro.set_title('Waveform  Comparison')
        ax_spectro.set_xlabel('Time Steps')
        ax_spectro.set_ylabel('Amplitude')
        
        # Plot 4: Denoising Performance
        ax_denoise = axes[1, 1]
        ax_denoise.set_title('Denoising Error')
        ax_denoise.set_xlabel('Time Steps')
        ax_denoise.set_ylabel('Error Amplitude')


        # Setup progress tracking
        mse_per_epoch = []
        train_losses = []
        val_losses = []

        BATCH_SIZE = 16
        # ADD THE START TIME
        start_time = perf_counter()

        # Calculate correct number of batches from input size
        total_samples = len(train_inputs)  # Get actual number of samples from input data
        print(f"Total number of samples: {total_samples}")  # Debug print

        num_batches_per_epoch = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
        total_batches = NUM_EPOCHS * num_batches_per_epoch

        print(f"Batches per epoch: {num_batches_per_epoch}")  # Debug print
        print(f"Total batches across all epochs: {total_batches}")  # Debug print

        # Create progress bars with proper positioning
        main_pbar = tqdm(total=total_batches, desc='Overall Progress', position=0)
        epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Epochs", position=1, leave=True)

        for epoch in epoch_pbar:
            epoch_mse = 0
            num_batches = 0
            
            # Process all samples in each epoch
            for i in range(0, total_samples, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, total_samples)
                
                # Debug prints for batch sizes
                print(f"Batch indices: {i}:{batch_end}")  # Debug print
                
                # Process batch inputs
                batch_inputs = train_inputs[i:batch_end]
                print(f"Batch inputs shape: {batch_inputs.shape}")  # Debug print
                
                flattened_inputs = batch_inputs.reshape(-1)
                input_ids = np.repeat(np.arange(NUM_INPUT), TIME_STEPS * len(batch_inputs))
                batch_spikes = preprocess_spikes(flattened_inputs, input_ids, NUM_INPUT)

                # # Fix the input_ids to ensure they stay within 0-63
                # input_ids = np.arange(NUM_INPUT)  # [0, 1, 2, ..., 63]
                # input_ids = np.tile(input_ids, TIME_STEPS * len(batch_inputs))  # Repeat pattern properly
                # # Now when we call the module's preprocess_spikes, it should work correctly
                # batch_spikes = preprocess_spikes(flattened_inputs, input_ids, NUM_INPUT)

                
                # Get corresponding clean target
                batch_outputs = train_outputs[i:i+1]  # Shape: (1, 45, 64)
                print(f"Batch outputs shape: {batch_outputs.shape}")  # Debug print
                
                # Train on batch
                metrics, cb_data = compiled_net.train(
                    {input_population: [batch_spikes]},
                    {output_population: batch_outputs},
                    num_epochs=1,
                    callbacks=callbacks
                )

                # Calculate loss
                output_values = cb_data["output_v"][-1]
                error = output_values - batch_outputs.reshape(-1, NUM_OUTPUT)
                batch_loss = np.mean(error * error)
                epoch_mse += batch_loss
                num_batches += 1
                
                # Update main progress bar
                main_pbar.update(1)
                main_pbar.set_postfix({
                    'Batch': f'{num_batches}/{num_batches_per_epoch}',
                    'MSE': f'{batch_loss:.6f}'
                })
            
            # Calculate epoch metrics
            avg_epoch_mse = epoch_mse / num_batches
            mse_per_epoch.append(avg_epoch_mse)
            train_losses.append(avg_epoch_mse)
            
            # Validation step
            val_loss = validate_network(compiled_net, val_inputs, val_outputs)
            val_losses.append(val_loss)
            
            # Update epoch progress
            epoch_pbar.set_postfix({
                'Train MSE': f'{avg_epoch_mse:.4f}',
                'Val MSE': f'{val_loss:.4f}'
            })

        # Close progress bars
        main_pbar.close()
        epoch_pbar.close()


        # Print final training summary
        total_time = perf_counter() - start_time    # Plot training progress
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Final Train MSE: {train_losses[-1]:.4f}")
        print(f"Final Val MSE: {val_losses[-1]:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', label='Training MSE', marker='o')
        plt.plot(val_losses, 'r-', label='Validation MSE', marker='o')
        plt.title('Final Training and Validation MSE History')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Use log scale if losses vary widely
        plt.savefig('final_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


                # Evaluate on test set
        print("\nEvaluating on test set...")
        test_mse, denoised_outputs = evaluate_on_test_set(compiled_net, test_inputs, test_outputs)
        print(f"Test Set MSE: {test_mse:.4f}")
        
        # Convert denoised_outputs to numpy array if it isn't already
        denoised_outputs = np.array(denoised_outputs)

        print("\nCalculating speech quality metrics...")
        all_test_metrics = []

        # Make sure we only process as many samples as we have
        num_samples = min(len(test_inputs), len(denoised_outputs))
        print(f"Processing {num_samples} test samples")

        # Calculate metrics for each test sample
        for i in range(num_samples):
            try:
                denoised = denoised_outputs[i].reshape(TIME_STEPS, NUM_INPUT)
                clean = test_outputs[i].reshape(TIME_STEPS, NUM_INPUT)
                metrics = calculate_speech_quality_metrics(denoised, clean)
                all_test_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                print(f"Denoised shape: {denoised_outputs[i].shape if hasattr(denoised_outputs[i], 'shape') else 'unknown'}")
                print(f"Clean shape: {test_outputs[i].shape}")
                continue

        # Print shapes for debugging
        print(f"\nShapes:")
        print(f"test_inputs shape: {test_inputs.shape}")
        print(f"test_outputs shape: {test_outputs.shape}")
        print(f"denoised_outputs shape: {denoised_outputs.shape if hasattr(denoised_outputs, 'shape') else len(denoised_outputs)}")

        # Calculate average metrics only if we have results
        if all_test_metrics:
            avg_metrics = {
                metric: np.mean([m[metric] for m in all_test_metrics])
                for metric in all_test_metrics[0].keys()
            }

            print("\nAverage Speech Quality Metrics across Test Set:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.2f}")
        else:
            print("No metrics were calculated successfully")

        # Plot distribution of metrics
        plt.figure(figsize=(15, 5))
        for idx, metric in enumerate(['SNR', 'PSNR', 'SSNR']):
            plt.subplot(1, 3, idx+1)
            values = [m[metric] for m in all_test_metrics]
            plt.hist(values, bins=20)
            plt.title(f'{metric} Distribution')
            plt.xlabel('Value (dB)')
            plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('speech_quality_metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot metrics over time for a single example
        plt.figure(figsize=(15, 5))
        example_idx = 0  # First test sample
        denoised = denoised_outputs[example_idx].reshape(TIME_STEPS, NUM_INPUT)
        clean = test_outputs[example_idx].reshape(TIME_STEPS, NUM_INPUT)

        # Calculate metrics for each time step
        time_metrics = []
        for t in range(TIME_STEPS):
            metrics = calculate_speech_quality_metrics(
                denoised[t:t+1], 
                clean[t:t+1]
            )
            time_metrics.append(metrics)

        # Plot metrics over time
        for metric in ['SNR', 'PSNR', 'SSNR']:
            values = [m[metric] for m in time_metrics]
            plt.plot(values, label=metric)
        plt.title('Speech Quality Metrics Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Value (dB)')
        plt.legend()
        plt.grid(True)
        plt.savefig('speech_quality_metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()


        # Modified final plot to include test results
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', label='Training MSE', marker='o')
        plt.plot(val_losses, 'r-', label='Validation MSE', marker='o')
        plt.axhline(y=test_mse, color='g', linestyle='--', label='Test MSE')
        plt.title('Training, Validation, and Test MSE History')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('final_training_history_with_test.png', dpi=300, bbox_inches='tight')
        plt.close()


         
        
        # Update visualizations
        # 1. MSE Learning Curve
        ax_mse.clear()
        ax_mse.plot(mse_per_epoch, 'b-', label='Training MSE')
        ax_mse.set_title('Training Progress')
        ax_mse.set_xlabel('Epoch')
        ax_mse.set_ylabel('MSE')
        ax_mse.grid(True)
        ax_mse.legend()


        time = np.arange(TIME_STEPS)
        
        # 3. Error Analysis (Bottom Left)
        
        error = denoised - clean
        error_mean = np.mean(error, axis=1)  # Average across frequency bands
        ax_spikes.plot(time, error_mean, 'r-', label='Error', alpha=0.7)
        ax_spikes.fill_between(time, error_mean, 0, alpha=0.3, color='red')
        ax_spikes.set_title('Denoising Error')
        ax_spikes.set_xlabel('Time Steps')
        ax_spikes.set_ylabel('Error Amplitude')
        ax_spikes.grid(True, alpha=0.3)

        denoised= cb_data["output_v"][-1].reshape(TIME_STEPS, NUM_INPUT) 
        # denoised = output_values
        
        # Plot frequency profiles
        # 1. Waveform Comparison (Top Left)
        sample_idx = 0  # Select first sample (or any other index you want to visualize)
        noisy_sample = train_inputs[sample_idx].reshape(TIME_STEPS, NUM_INPUT)  # Shape: (45, 64)
        clean_sample = train_outputs[sample_idx].reshape(TIME_STEPS, NUM_INPUT)  # Shape: (45, 64)

        # Now plot with correct dimensions
        ax_spectro.plot(time, np.mean(noisy_sample, axis=1), 'r-', label='Noisy', alpha=0.6, linewidth=1)
        ax_spectro.plot(time, np.mean(clean_sample, axis=1), 'g-', label='Clean', alpha=0.6, linewidth=1)
        ax_spectro.plot(time, np.mean(denoised, axis=1), 'b-', label='Denoised', alpha=0.6, linewidth=1)
        ax_spectro.set_title('Waveform Comparison')
        ax_spectro.set_xlabel('Time Steps')
        ax_spectro.set_ylabel('Average Amplitude')
        ax_spectro.legend()
        ax_spectro.grid(True, alpha=0.3)
        
        
        
        # 3. Neural Activity
        ax_spikes.clear()
        input_spikes = cb_data["input_spikes"]
        hidden_spikes = cb_data["hidden_spikes"]
        ax_spikes.scatter(input_spikes[0][-1], input_spikes[1][-1], 
                        c='blue', s=1, alpha=0.5, label='Input')
        ax_spikes.scatter(hidden_spikes[0][-1], hidden_spikes[1][-1], 
                        c='red', s=1, alpha=0.5, label='Hidden')
        ax_spikes.legend()
        ax_spikes.set_title('Neural Activity')



        ax_denoise.clear()
        denoised_spec = denoised.T
        im = ax_denoise.imshow(denoised_spec, 
                            aspect='auto', 
                            origin='lower', 
                            cmap='viridis')

        # Add colorbar
        plt.colorbar(im, ax=ax_denoise, label='Amplitude')

        # Enhance the plot
        ax_denoise.set_title('Denoising Performance (All Frequencies)')
        ax_denoise.set_xlabel('Time Steps')
        ax_denoise.set_ylabel('Frequency Bands')

        # Add frequency band markers
        freq_ticks = np.linspace(0, NUM_INPUT-1, 5)
        freq_labels = [f'F{int(f)}' for f in freq_ticks]
        ax_denoise.set_yticks(freq_ticks)
        ax_denoise.set_yticklabels(freq_labels)

        # Add grid
        ax_denoise.grid(True, alpha=0.3)


        # ax_denoise.clear()
        # ax_denoise.plot(denoised_spec, aspect='auto', origin='lower', cmap='viridis')
        # ax_denoise.set_title('Denoised Output')
        # ax_denoise.set_xlabel('Time Steps')
        # ax_denoise.set_ylabel('Frequency Bands')
        # ax_denoise.colorbar(im_error, ax=ax_denoise)
        # ax_denoise.grid(True, alpha=0.3)


        

        # ax_denoise.clear()
        # band_idx = NUM_INPUT // 2  # Middle frequency band
        # ax_denoise.plot(time, noisy_sample[:, band_idx], 'r-', alpha=0.5, label='Noisy')
        # ax_denoise.plot(time, clean_sample[:, band_idx], 'g--', alpha=0.5, label='Clean')
        # ax_denoise.plot(time, denoised[:, band_idx], 'b-', alpha=0.5, label='Denoised')
        # ax_denoise.legend()
        # ax_denoise.set_title('Denoising Performance (Mid Frequency Band)')
        
        # Update the figure
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update the display


        end_time = perf_counter()
        print(f"Training time: {end_time - start_time:.2f}s")
            
        
        # # Early stopping check
        # if epoch_mse < best_mse:
        #     best_mse = epoch_mse
        #     patience_counter = 0
        #     # Here you could save the best model if needed
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        #         break
        
        # Print progress with more details
        # print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        # print(f"MSE: {epoch_mse:.4f}")
        # print(f"Best MSE: {best_mse:.4f}")
        # print(f"Patience Counter: {patience_counter}/{patience}")
        
        # # Update progress bar with cleaner postfix
        # pbar.set_postfix_str(
        #     f"MSE: {epoch_mse:.4f}, Best: {best_mse:.4f}, Patience: {patience_counter}/{patience}"
        # )
    
    # Final save
    fig.savefig('FINAL_training_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save MSE history separately
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_per_epoch) + 1), mse_per_epoch, 'b-', marker='o')
    plt.title('Training MSE History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig('mse_history.png', dpi=300, bbox_inches='tight')
    plt.close()



            # Replace your final plotting code (around line 565)
    # Final plot at the end of training


        # 4. Clear and Comparative Denoising Performance
    ax_denoise.clear()

    # Create a figure with three subplots side by side
    fig_denoise = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig_denoise, wspace=0.3)

    # 1. Noisy Input
    ax1 = fig_denoise.add_subplot(gs[0])
    im1 = ax1.imshow(noisy_sample.T, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis')
    ax1.set_title('Noisy Input', fontsize=12, pad=10)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Frequency Bands')
    plt.colorbar(im1, ax=ax1, label='Amplitude')

    # 2. Clean Target
    ax2 = fig_denoise.add_subplot(gs[1])
    im2 = ax2.imshow(clean_sample.T, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis')
    ax2.set_title('Clean Target', fontsize=12, pad=10)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Frequency Bands')
    plt.colorbar(im2, ax=ax2, label='Amplitude')

    # 3. Network Output (Denoised)
    ax3 = fig_denoise.add_subplot(gs[2])
    im3 = ax3.imshow(denoised.T, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis')
    ax3.set_title('Network Output (Denoised)', fontsize=12, pad=10)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Frequency Bands')
    plt.colorbar(im3, ax=ax3, label='Amplitude')

    # Add frequency labels to all plots
    freq_ticks = np.linspace(0, NUM_INPUT-1, 5, dtype=int)
    freq_labels = [f'F{f}' for f in freq_ticks]
    for ax in [ax1, ax2, ax3]:
        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels)
        
        # Add time ticks
        time_ticks = np.linspace(0, TIME_STEPS-1, 5, dtype=int)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks)

    # Add a main title
    fig_denoise.suptitle('Speech Denoising Performance Comparison', 
                        fontsize=14, y=1.05)

    # Save the figure
    plt.savefig(f'denoising_comparison_epoch.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig_denoise)

    # Update the main visualization's denoising subplot with error analysis
    ax_denoise.clear()
    error = denoised.T - clean_sample.T
    im_error = ax_denoise.imshow(error, 
                                aspect='auto',
                                origin='lower',
                                cmap='RdBu_r',  # Red-Blue diverging colormap
                                vmin=-np.max(abs(error)),
                                vmax=np.max(abs(error)))
    ax_denoise.set_title('Denoising Error\n(Blue: Removed Noise, Red: Added Noise)')
    ax_denoise.set_xlabel('Time Steps')
    ax_denoise.set_ylabel('Frequency Bands')
    plt.colorbar(im_error, ax=ax_denoise)
        



    
            
    # Print final losses
    print(f"\nFinal Training MSE: {train_losses[-1]:.4f}")
    print(f"Final Validation MSE: {val_losses[-1]:.4f}")












