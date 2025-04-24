import os
import numpy as np
import torch
from itertools import product
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import SpikeRecorder, VarRecorder, OptimiserParamSchedule
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
import gc
import psutil
from time import perf_counter
from ml_genn.compilers.event_prop_compiler import default_params
from collections import defaultdict
import random
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Import necessary functions from part2
from part2 import (
    compute_mel,
    apply_percentile_threshold,
    preprocess_spikes
)

# Constants
NUM_INPUT = 64
NUM_OUTPUT = 64
NUM_HIDDEN = 256  # Default value
TAU_SYN = 5.0    # Default value
TIME_STEPS = 45
MAX_SPIKES = TIME_STEPS * NUM_INPUT * 5060

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class SpeechSNNTuner:
    def __init__(self, seed=42):
        # Set random seeds
        set_random_seeds(seed)
        
        # Define search space specific to speech denoising SNN
        self.space = {
            # Network architecture - using explicit values instead of indices
            'NUM_HIDDEN': hp.choice('NUM_HIDDEN', [128, 256, 512]),
            
            # Neuron parameters
            'V_THRESH': hp.uniform('V_THRESH', 0.5, 1.2),
            'TAU_MEM': hp.uniform('TAU_MEM', 15.0, 25.0),
            'TAU_SYN': hp.uniform('TAU_SYN', 5.0, 15.0),
            
            # Training parameters - using explicit values
            'BATCH_SIZE': hp.choice('BATCH_SIZE', [8, 16, 32]),
            'LEARNING_RATE': hp.loguniform('LEARNING_RATE', np.log(1e-4), np.log(1e-2)),
            'NUM_EPOCHS': hp.choice('NUM_EPOCHS', [10, 30, 50])
        }
        
        # Add these new attributes
        self.train_losses = []
        self.val_losses = []
        self.trials = Trials()
        self.best_result = None
        self.best_trial_losses = None

                # Load and prepare data
        self.prepare_data()
        
    def objective(self, params):
        """Objective function with memory management"""
        try:
            # Clear memory before each trial
            gc.collect()
            torch.cuda.empty_cache()
            
            # Train network with current parameters
            train_losses, val_losses, compiled_net = self.train_network(params)
            
            # Store the losses
            self.train_losses = train_losses
            self.val_losses = val_losses
            
            final_loss = val_losses[-1]
            
            # Clean up network
            del compiled_net
            gc.collect()
            
            return {
                'loss': final_loss,
                'status': STATUS_OK,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'params': params
            }
            
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return {'status': 'fail', 'loss': float('inf')}
    
    def run_optimization(self, max_evals=30):
        """Run optimization with memory management"""
        try:
            print("Starting optimization with full dataset...")
            self.prepare_data(max_files=5060)
            
            # Add memory cleanup
            gc.collect()
            torch.cuda.empty_cache()  # If using GPU
            
            best = fmin(
                fn=self.objective,
                space=self.space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=self.trials
            )

                # Store and save results
            self.best_result = best
            self.save_final_results()
            
            return best
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None
        
    
   
    
    def save_final_results(self):
        """Save and display final optimization results with plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        
        if self.best_result:
            # Map indices to actual values for choice parameters
            param_choices = {
                'NUM_HIDDEN': [128, 256, 512],
                'BATCH_SIZE': [8, 16, 32],
                'NUM_EPOCHS': [10, 30, 50]
            }
            
            # Convert numpy types and map choices to actual values
            best_params = {}
            for param, value in self.best_result.items():
                if param in param_choices:
                    # Convert index to actual value for choice parameters
                    value = param_choices[param][int(value)]
                elif isinstance(value, (np.int32, np.int64)):
                    value = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    value = float(value)
                best_params[param] = value
            
            # Display best parameters
            print("\nBest Parameters Found:")
            for param, value in best_params.items():
                print(f"{param:<15} : {value}")
            
            # Find best trial
            best_trial = min(self.trials.trials, key=lambda t: t['result']['loss'])
            
            # Create plots
            # 1. Learning Curves
            plt.figure(figsize=(10, 5))
            plt.plot(best_trial['result']['train_losses'], label='Training Loss')
            plt.plot(best_trial['result']['val_losses'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'learning_curves_{timestamp}.png')
            plt.close()
            
            # 2. Speech SNN Results
            plt.figure(figsize=(10, 5))
            trial_losses = [t['result']['loss'] for t in self.trials.trials 
                           if t['result']['status'] == STATUS_OK]
            plt.plot(trial_losses, 'b.-')
            plt.xlabel('Trial')
            plt.ylabel('Loss')
            plt.title('Speech SNN Optimization Progress')
            plt.grid(True)
            plt.savefig(f'speech_snn_results_{timestamp}.png')
            plt.close()
            
            # 3. Hyperparameter Results
            plt.figure(figsize=(12, 6))
            param_values = defaultdict(list)
            losses = []
            
            for trial in self.trials.trials:
                if trial['result']['status'] == STATUS_OK:
                    for param, value in trial['misc']['vals'].items():
                        param_values[param].extend(value)
                    losses.append(trial['result']['loss'])
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, (param, values) in enumerate(param_values.items()):
                if idx < len(axes):
                    axes[idx].scatter(values, losses)
                    axes[idx].set_xlabel(param)
                    axes[idx].set_ylabel('Loss')
                    axes[idx].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'hyperparameter_results_{timestamp}.png')
            plt.close()
            
            # Create results dictionary with converted types
            results = {
                'timestamp': timestamp,
                'best_parameters': best_params,
                'best_trial': {
                    'loss': float(best_trial['result']['loss']),
                    'train_losses': [float(x) for x in best_trial['result'].get('train_losses', [])],
                    'val_losses': [float(x) for x in best_trial['result'].get('val_losses', [])]
                },
                'optimization_summary': {
                    'num_trials': len(self.trials.trials),
                    'best_loss': float(best_trial['result']['loss']),
                    'trial_losses': [float(t['result']['loss']) for t in self.trials.trials 
                                   if t['result']['status'] == STATUS_OK]
                }
            }
            
            # Save to file
            with open(f'snn_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"\nResults saved:")
            print(f"- snn_results_{timestamp}.json")
            print(f"- learning_curves_{timestamp}.png")
            print(f"- speech_snn_results_{timestamp}.png")
            print(f"- hyperparameter_results_{timestamp}.png")





    def prepare_data(self, max_files=5060):
        """Load and prepare the dataset"""
        # Clear memory before starting
        gc.collect()

        # Directories
        speech_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/preprocessed_speech"
        output_dir = "/its/home/msu22/attempt_DISS/STAGE1/NEW_PART/comb_speech"

        # Compute mel spectrograms
        print("Computing mel spectrograms...")
        comb_mel_results = compute_mel(output_dir, max_files=max_files)
        clean_mel_results = compute_mel(speech_dir, max_files=max_files)

        # Prepare inputs and outputs
        comb_inputs = []
        clean_outputs = []

        # Process clean outputs first
        print("Processing clean outputs...")
        for file in clean_mel_results.keys():
            # Clean outputs are already normalized from compute_mel
            clean_outputs.append(clean_mel_results[file].numpy())
        
        gc.collect()  # Clear memory after processing outputs

        # Process combined inputs
        print("Processing combined inputs...")
        for file in comb_mel_results.keys():
            # Apply thresholding to normalized input
            thresholded_input = apply_percentile_threshold(comb_mel_results[file].numpy(), percentile=35)
            comb_inputs.append(thresholded_input)

        gc.collect()  # Clear memory after processing inputs

        # Convert to numpy arrays and store as class attributes
        self.inputs = np.array(comb_inputs)
        self.outputs = np.array(clean_outputs)

        print(f"Input shape: {self.inputs.shape}")
        print(f"Output shape: {self.outputs.shape}")

        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        train_val_inputs, self.test_inputs, train_val_outputs, self.test_outputs = train_test_split(
            self.inputs, self.outputs, test_size=0.2, random_state=42
        )

        # Second split: separate train and validation
        self.train_inputs, self.val_inputs, self.train_outputs, self.val_outputs = train_test_split(
            train_val_inputs, train_val_outputs, test_size=0.1, random_state=42
        )

        print(f"Training samples: {len(self.train_inputs)}")
        print(f"Validation samples: {len(self.val_inputs)}")
        print(f"Test samples: {len(self.test_inputs)}")

        # Final memory cleanup
        gc.collect()

    def alpha_schedule(self, epoch, alpha):
        """Learning rate schedule function"""
        if (epoch % 2) == 0 and epoch != 0:
            return alpha * 0.7
        else:
            return alpha
            
    def train_network(self, params):
        """Train network with given parameters"""

        gc.collect()
        # Create network
        network = Network(default_params)
        with network:
            # Populations
            input_population = Population(
                SpikeInput(max_spikes=MAX_SPIKES),
                NUM_INPUT,
                record_spikes=True
            )
            
            hidden_population = Population(
                LeakyIntegrateFire(
                    v_thresh=params['V_THRESH'],
                    tau_mem=params['TAU_MEM'],
                    tau_refrac=None
                ),
                params['NUM_HIDDEN'],
                record_spikes=True
            )
            
            output_population = Population(
                LeakyIntegrate(
                    tau_mem=params['TAU_MEM'],
                    readout="var"
                ),
                NUM_OUTPUT,
                record_spikes=True
            )
            
            # Connections
            Connection(
                input_population,
                hidden_population,
                Dense(Normal(sd=2.5 / np.sqrt(NUM_INPUT))),
                Exponential(params['TAU_SYN'])
            )
            
            Connection(
                hidden_population,
                hidden_population,
                Dense(Normal(sd=1.5 / np.sqrt(params['NUM_HIDDEN']))),
                Exponential(params['TAU_SYN'])
            )
            
            Connection(
                hidden_population,
                output_population,
                Dense(Normal(sd=2.0 / np.sqrt(params['NUM_HIDDEN']))),
                Exponential(params['TAU_SYN'])

            )

        gc.collect()

        # Create callbacks
        callbacks = [
            VarRecorder(output_population, "v", key="output_v"),
            SpikeRecorder(input_population, key="input_spikes"),
            SpikeRecorder(hidden_population, key="hidden_spikes"),
            OptimiserParamSchedule("alpha", self.alpha_schedule)
        ]

        # Compile network
        compiler = EventPropCompiler(
            example_timesteps=TIME_STEPS,
            losses="mean_square_error",
            optimiser=Adam(params['LEARNING_RATE']),
            reg_lambda_upper=1e-6,
            reg_lambda_lower=1e-6,
            reg_nu_upper=15,
            max_spikes=MAX_SPIKES
        )
        compiled_net = compiler.compile(network)

        # Use compiled network within context manager
        with compiled_net:
                # Setup progress tracking
            mse_per_epoch = []
            train_losses = []
            val_losses = []

            BATCH_SIZE = params['BATCH_SIZE']
            # ADD THE START TIME
            start_time = perf_counter()

            # Calculate correct number of batches from input size
            total_samples = len(self.train_inputs)  # Get actual number of samples from input data
            print(f"Total number of samples: {total_samples}")  # Debug print

            num_batches_per_epoch = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
            total_batches = params['NUM_EPOCHS'] * num_batches_per_epoch

            print(f"Batches per epoch: {num_batches_per_epoch}")  # Debug print
            print(f"Total batches across all epochs: {total_batches}")  # Debug print

            # Create progress bars with proper positioning
            main_pbar = tqdm(total=total_batches, desc='Overall Progress', position=0)
            epoch_pbar = tqdm(range(params['NUM_EPOCHS']), desc="Epochs", position=1, leave=True)

            for epoch in epoch_pbar:
                epoch_mse = 0
                num_batches = 0
                
                
                # Process all samples in each epoch
                for i in range(0, total_samples, BATCH_SIZE):
                    batch_end = min(i + BATCH_SIZE, total_samples)
                    
                    # Debug prints for batch sizes
                    print(f"Batch indices: {i}:{batch_end}")  # Debug print
                    
                    # Process batch inputs
                    batch_inputs = self.train_inputs[i:batch_end]
                    # Transpose inputs if needed: (16, 64, 45) -> (16, 45, 64)
                    batch_inputs = np.transpose(batch_inputs, (0, 2, 1))
                    
                    flattened_inputs = batch_inputs.reshape(-1)
                    input_ids = np.repeat(np.arange(NUM_INPUT), TIME_STEPS * len(batch_inputs))
                    batch_spikes = preprocess_spikes(flattened_inputs, input_ids, NUM_INPUT)
                    
                    # Get corresponding clean target and transpose
                    batch_outputs = self.train_outputs[i:i+1]  # Shape: (1, 64, 45)
                    # Transpose outputs: (1, 64, 45) -> (1, 45, 64)
                    batch_outputs = np.transpose(batch_outputs, (0, 2, 1))
                    
                    # Verify shapes
                    print(f"Transposed batch outputs shape: {batch_outputs.shape}")  # Should be (1, 45, 64)
                    
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
                val_loss = self.validate_network(compiled_net, params, input_population, output_population)
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

        return train_losses, val_losses, compiled_net

    def validate_network(self, compiled_net, params, input_population, output_population):
        """Run validation"""
        val_loss = 0
        num_batches = 0

        # Create callbacks for validation
        spike_recorder = SpikeRecorder(input_population, key="input_spikes")
        var_recorder = VarRecorder(output_population, "v", key="output_v")
        callbacks = [spike_recorder, var_recorder]

        for i in range(0, len(self.val_inputs), params['BATCH_SIZE']):
            batch_losses = []
            batch_end = min(i + params['BATCH_SIZE'], len(self.val_inputs))
            
            for j in range(i, batch_end):
                # Process single input
                single_input = self.val_inputs[j:j+1]
                flattened_input = single_input.reshape(1, TIME_STEPS, NUM_INPUT)
                
                # Now flatten for spike processing
                spike_values = flattened_input.reshape(-1)
                input_ids = np.repeat(np.arange(NUM_INPUT), TIME_STEPS * len(single_input))

                # Process spikes
                batch_spikes = preprocess_spikes(spike_values, input_ids, NUM_INPUT)
                
                # Process single output - ensure shape is (1, 45, 64)
                batch_outputs = self.val_outputs[j:j+1].reshape(1, TIME_STEPS, NUM_OUTPUT)
                
                # Validate single sample
                metrics , cb_data = compiled_net.train(
                    {input_population: [batch_spikes]},
                    {output_population: batch_outputs},
                    num_epochs=1,
                    callbacks=callbacks
                )
                
                output_values = cb_data["output_v"][-1]
                error = output_values - batch_outputs.reshape(-1, NUM_OUTPUT)
                sample_loss = np.mean(error * error)
                batch_losses.append(sample_loss)
    
            val_loss += np.mean(batch_losses)
            num_batches += 1

        return val_loss / num_batches

    def run_grid_search(self):
        """Run grid search over parameter combinations"""
        param_names = list(self.hyperparams.keys())
        param_values = list(self.hyperparams.values())
        
        # Calculate total combinations
        total_combinations = np.prod([len(values) for values in param_values])
        print(f"Total parameter combinations to try: {total_combinations}")
        
        # Progress bar for combinations
        pbar = tqdm(total=total_combinations, desc="Parameter combinations")
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            print(f"\nTesting parameters: {params}")
            
            train_losses, val_losses, compiled_net = self.train_network(params)
            
            # Store results
            result = {
                'params': params,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1]
            }
            
            self.results.append(result)
            
            # Update best parameters if needed
            if val_losses[-1] < self.best_score:
                self.best_score = val_losses[-1]
                self.best_params = params
            
            # Save intermediate results
            self.save_results()
            
            # Plot learning curves
            self.plot_learning_curves(result)
            
            pbar.update(1)
        
        pbar.close()

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparam_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'results': self.results,
                'best_params': self.best_params,
                'best_score': self.best_score
            }, f, indent=4)

    def plot_training_results(self):
        """Plot all training-related visualizations"""
        if not hasattr(self, 'train_losses') or not self.train_losses:
            print("No training data available to plot")
            return
            
        plt.style.use('seaborn-v0_8')  # Updated style name
        
        # 1. Training and Validation Loss Curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_validation_loss.png')
        plt.close()

        # 2. Loss vs Trial Number
        plt.figure(figsize=(12, 6))
        trial_losses = [t['result']['loss'] for t in self.trials.trials if t['result']['status'] == 'ok']
        trial_numbers = range(1, len(trial_losses) + 1)
        plt.plot(trial_numbers, trial_losses, 'b-', marker='o')
        plt.title('Loss vs Trial Number')
        plt.xlabel('Trial Number')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.savefig('loss_vs_trial.png')
        plt.close()

        # 3. Learning Rate vs Loss
        plt.figure(figsize=(12, 6))
        learning_rates = [t['misc']['vals']['LEARNING_RATE'][0] 
                        for t in self.trials.trials if t['result']['status'] == 'ok']
        plt.semilogx(learning_rates, trial_losses, 'ro')
        plt.title('Learning Rate vs Loss')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.savefig('learning_rate_vs_loss.png')
        plt.close()

        # 4. Batch Size vs Loss
        plt.figure(figsize=(12, 6))
        batch_sizes = [t['misc']['vals']['BATCH_SIZE'][0] 
                    for t in self.trials.trials if t['result']['status'] == 'ok']
        plt.plot(batch_sizes, trial_losses, 'go')
        plt.title('Batch Size vs Loss')
        plt.xlabel('Batch Size')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.savefig('batch_size_vs_loss.png')
        plt.close()

        # 5. Best Trial Learning Curves
        best_trial_idx = np.argmin(trial_losses)
        best_train_losses = self.trials.trials[best_trial_idx]['result'].get('train_losses', [])
        best_val_losses = self.trials.trials[best_trial_idx]['result'].get('val_losses', [])
        
        if best_train_losses and best_val_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(best_train_losses, label='Training Loss', color='blue')
            plt.plot(best_val_losses, label='Validation Loss', color='red')
            plt.title('Best Trial Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig('best_trial_learning_curves.png')
            plt.close()




if __name__ == "__main__":
    # Set fixed seed
    set_random_seeds(42)
    
    # Create tuner with same seed
    tuner = SpeechSNNTuner(seed=42)
    
    # Run optimization
    best_params = tuner.run_optimization()
    
    # Convert NumPy types to Python native types
    if best_params:
        converted_params = {}
        for key, value in best_params.items():
            if isinstance(value, np.integer):
                converted_params[key] = int(value)
            elif isinstance(value, np.floating):
                converted_params[key] = float(value)
            else:
                converted_params[key] = value
    
        # Print best parameters
        print("\nBest parameters found:")
        print(json.dumps(converted_params, indent=4))
    
    # Plot results
    print("Plotting training results...")
    tuner.plot_training_results()