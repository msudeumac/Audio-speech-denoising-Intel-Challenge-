# Speech Denoising with Spiking Neural Networks (SNNs)

This project explores the application of **Spiking Neural Networks (SNNs)** for **speech denoising**, implementing multiple approaches to improve the quality of speech signals in noisy environments. The project includes comprehensive analysis tools, visualization capabilities, and different implementation strategies.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Implementation Approaches](#implementation-approaches)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Visualizations](#results-and-visualizations)
- [License](#license)

## Project Overview

This research project focuses on speech denoising using Spiking Neural Networks, featuring multiple implementation approaches and comprehensive analysis tools. The project includes preprocessing pipelines, different neural network architectures, and visualization tools for analyzing the results.

## Project Structure

```
├── Approach1.py              # First implementation approach
├── Approach2.py              # Second implementation approach
├── Pre-processing.py         # Data preprocessing pipeline
├── Stats.py                  # Statistical analysis tools
├── hyperparameter.py         # Hyperparameter optimization
├── mel_spec.py              # Mel spectrogram generation
├── Graphs_mel.py            # Mel spectrogram visualization
├── preprocessed_speech/      # Processed speech data
├── preprocessed_noise/       # Processed noise data
└── comb_speech/             # Combined speech datasets
```

## Implementation Approaches

### Approach 1
The first implementation strategy focuses on [brief description of approach 1's unique features]
- Implementation details in `Approach1.py`
- Key features and methodology

### Approach 2
The second implementation approach explores [brief description of approach 2's unique features]
- Implementation details in `Approach2.py`
- Different methodology and results

## Features

- **Comprehensive Preprocessing Pipeline**: Robust preprocessing of speech and noise data
- **Multiple Implementation Approaches**: Two different strategies for speech denoising
- **Advanced Visualization Tools**: 
  - Mel spectrogram analysis and visualization
  - Distribution comparisons
  - Denoising performance visualization
  - Spike filtering analysis
- **Statistical Analysis**: Detailed statistical tools for performance evaluation

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing Data
```bash
python Pre-processing.py
```

### Running Different Approaches
```bash
# For Approach 1
python Approach1.py

# For Approach 2
python Approach2.py
```

### Generating Visualizations
```bash
# For mel spectrogram visualization
python Graphs_mel.py

# For statistical analysis
python Stats.py
```

## Results and Visualizations

The project includes several visualization outputs:

- `FINAL_training_visualization.png`: Final training results
- `mel_distributions_comparison.png`: Comparison of mel spectrogram distributions
- `denoising_comparison_epoch.png`: Epoch-wise denoising performance
- `spike_filtering_comparison.png`: Spike filtering analysis
- `threshold_comparison.png`: Threshold analysis results

## License

This project is licensed under the terms included in the LICENSE file.

