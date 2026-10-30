# Denoising Neural Processes (DnNP)


## Overview

Neural Processes (NPs) learn distributions over functions from context points, but they assume the input coordinates of those points are known exactly. In real sensing systems, such as low-cost air quality sensors, weather stations, and mobile platforms, the locations and timestamps attached to each measurement are themselves noisy.

**DnNP** extends Neural Processes to handle **noisy input**. Using variational inference, the model treats the true inputs as latent variables and learns to denoise them jointly with the function it predicts, producing calibrated uncertainty that accounts for noise in both the inputs and the outputs.

<img width="1656" height="652" alt="image" src="https://github.com/user-attachments/assets/a466339c-6937-4ebf-b162-0c66ad365293" />

## Key Contributions

- A Neural Process formulation that models uncertainty in input, not only in outputs.
- A variational inference scheme that jointly infers denoised inputs and the latent function.
- Experiments on synthetic and real-world data showing improved predictive accuracy and likelihood under input noise.

## Datasets

We evaluate DnNP on four datasets:

| Dataset | Type | Description |
|---|---|---|
| **Toy (synthetic)** | 1D regression | Functions sampled with controlled noise added to the input coordinates. |
| **MNIST** | Image completion | Pixel-coordinate noise added to context points for 2D image completion. |
| **iSUPER** | Air quality | Hyperlocal air quality measurements from a dense low-cost sensor network. [Access the data here](ISUPER_DATA_LINK). |
| **NOAA** | Air Pressure | Station-based meteorological observations from NOAA. [Source](NOAA_DATA_LINK) |

<img width="1716" height="1106" alt="image" src="https://github.com/user-attachments/assets/dac5cc91-c4fc-448e-9b2d-3dd5da351219" />

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tohidian2026dnnp,
  title     = {DnNP: Denoising Input Uncertainty in Neural Processes},
  author    = {Fatemeh Tohidian, Chengzhi Shi, Soomi Lee, Matthew Elia, Amy Mueller, Stratis Ioannidis Jennifer Dy},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Contact

Fatemeh Tohidian — [fatemeh.tohidian79@gmail.com](mailto:fatemeh.tohidian79@gmail.com)
