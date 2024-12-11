# CogVideoX: Text-to-Video Generation with Enhanced VAE Architectures

## Overview

This repository is dedicated to the research and development of Variational Autoencoders (VAEs) for text-to-video generation, focusing on improving spatial and temporal coherence in generated videos. The work was part of a collaborative effort by a team of researchers at Arizona State University, including significant contributions by **Abhiram Vadlapatla**.

## Key Contributions by Abhiram Vadlapatla

### Dataset Preprocessing and Analysis
- **Sample Selection**: Curated a subset of 2000 diverse video clips from the OpenVid-1M dataset to ensure robust benchmarking.
- **Feature Analysis**: Conducted detailed visualizations of key dataset properties such as camera motion, aesthetic scores, temporal consistency, and video duration.
- **Data Preprocessing Pipeline**: Developed a stratified sampling mechanism to ensure balanced representation of diverse video attributes, such as motion and aesthetic scores.

### Implementation and Optimization
- **Encoding-Decoding Pipeline**: Implemented the video encoding and decoding process using `diffusers.AutoencoderKLCogVideoX`. Optimized GPU memory usage with slicing, tiling, and mixed-precision techniques to accommodate computational constraints.
- **Evaluation Metrics**: Computed metrics such as SSIM (Structural Similarity Index Measure) and PSNR (Peak Signal-to-Noise Ratio) to assess reconstruction quality. 
- **Visualization**: Generated visual representations of the original, latent, and reconstructed frames to validate the model's performance qualitatively.

### Results and Analysis
- **Metric Insights**: Highlighted the high SSIM scores (> 0.9) across most videos, indicating strong structural preservation, and PSNR values concentrated in the 32–35 dB range, demonstrating minimal reconstruction distortion.
- **Correlation Analysis**: Identified key relationships between metrics such as SSIM, PSNR, and temporal consistency, emphasizing the alignment between spatial fidelity and temporal coherence.

### Challenges Addressed
- **Memory Constraints**: Implemented advanced memory management strategies, including manual cache clearing and garbage collection, to handle large-scale video processing.
- **Error Handling**: Designed robust error-handling mechanisms to ensure smooth processing despite corrupted files or unsupported formats.

## Highlights of the Project

1. **Advanced VAE Architectures**: Explored variations in spatial and temporal compression to optimize performance.
2. **Benchmarking and Metrics**: Delivered comprehensive evaluation using state-of-the-art quality measures like SSIM and PSNR.
3. **OpenVid-1M Dataset**: Leveraged a diverse dataset to drive innovation in text-to-video generation.

## Technologies Used
- **Programming Languages**: Python
- **Frameworks**: PyTorch, Hugging Face Diffusers
- **Tools**: Matplotlib, scikit-learn, scikit-image, Pandas
- **Hardware**: NVIDIA A100 GPUs with 80GB memory

## Getting Started

### Prerequisites
- Python 3.8 or later
- PyTorch and associated libraries
- Hugging Face Diffusers

### Installation
Clone this repository and follow the setup instructions and run the vae_psnr file to run the experiments for the vae.

## Acknowledgments
This repository is part of a collaborative project as part of CSE 598: Frontier topics in Gen AI.
