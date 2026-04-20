# Structure-Adaptive Sparse Diffusion in Voxel Space for 3D Medical Image Enhancement

<font size=3><div align='center' > **Paper (Coming Soon)** | [**Datasets**](#datasets) | [**Model**](#model) | [**Training**](#training) | [**Inference**](#inference)</div></font>

Official PyTorch implementation of the paper above.


## Requirements Installation
First, clone the repository to your local machine:
```bash
git clone https://github.com/mirthAI/sparse-3d-diffusion.git
cd sparse-3d-diffusion
```
To install the required packages, you can use the following command:
```bash
conda env create -f env.yaml
conda activate sparse-3d-diffusion
```
or
```bash
pip install -r requirements.txt
```

## Datasets
In the paper, we train and evaluate our model on two 3D enhancement tasks: image denoising and super-resolution using four datasets.

 Dataset  | Modality | Volumes | Download Link |
| ------------- | ------------- | ------------- | ------------- |
| LDCT-and-Projection-data Dataset | CT |	50 | Coming Soon |
| FDG PET Dataset | PET |	88 | Coming Soon |
| AortaSeg24 Dataset | CTA |	60 | Coming Soon |
| UHB FCD Lesion Dataset | MRI | 120 | Coming Soon |

The datasets should be downloaded and placed in the `data` folder of the project. 

## Model

| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Lung CT Denoising | Coming Soon |
| Brain PET Denoising | Coming Soon |
| Aorta CTA SR | Coming Soon |
| Brain MRI SR | Coming Soon |

## Training
To start training the model, use the following command:

```bash
sh slurm_script/shell_train_2G.sh --config <YOUR CONFIG>
```

You can resume training from saved checkpoints by setting "--resume_checkpoint"

## Inference
Before inference, download the pretrained model from the [Model](#model) section and place it under `checkpoints/`. Then run:

```bash
sh slurm_script/shell_sample_2G.sh --config <YOUR CONFIG>
```

## References
The code is mainly adapted from [Guided-Diffusion](https://github.com/openai/guided-diffusion).


## Citations and Acknowledgements
The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Hongxu Jiang at hongxu.jiang@ufl.edu.

Kindly cite the following papers if you use our code.

```bibtex
@article{jiang2026structure,
  title={Structure-Adaptive Sparse Diffusion in Voxel Space for 3D Medical Image Enhancement},
  author={Jiang, Hongxu and others},
  journal={Coming Soon},
  year={2026}
}
```
