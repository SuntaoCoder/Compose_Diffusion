# Composable Diffusion Model for Generating Crystal Structures with Multi-Target Properties

This code is improved on the basis of [MatterGen](https://www.nature.com/articles/s41586-025-08628-5), and achieved multi-target property crystal generation.


## Developers

Tao Sun

## Usage

 ```bash
 python scripts/compose_generate.py --output_path="results/output_mag0.05_bandgap1" --pretrained-name "['dft_mag_density', 'dft_band_gap']" --batch_size=16 --num_batches=1 --properties_to_condition_on="{'dft_mag_density': 0.05, 'dft_band_gap': 1.0}" --diffusion_guidance_factor=2.0
```