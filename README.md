# Hypergraph Neural Sheaf Diffusion (HNSD)

This repository contains the official implementation of the paper:

**"Hypergraph Neural Sheaf Diffusion: A Symmetric Simplicial Set Framework for Higher-Order Learning"** 

---

## Requirements
- matplotlib>=3.5
- numpy>=1.21
- scikit-learn>=1.0
- scipy>=1.7
- torch>=1.13
- torch-geometric>=1.6
- torch-scatter>=2.1
- torch-sparse>=0.6
- tqdm>=4.64
- wandb>=0.13

---

## Usage

### Training
```bash
cd src
python train.py --dname <DATASET_NAME> --method SheafHyperGNNDiag --new_edge True --earlystop True --plot True
```

## Dataset Preparation
To prepare the datasets, follow the guidelines in the [AllSet repository](https://github.com/jianhao2016/AllSet).  
Once generated, move the `data` folder into `../data`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Choi_2025,
  title     = {Hypergraph Neural Sheaf Diffusion: A Symmetric Simplicial Set Framework for Higher-Order Learning},
  author    = {Choi, Seongjin and Kim, Gahee and Oh, Yong-Geun},
  journal   = {IEEE Access},
  volume    = {13},
  pages     = {131823--131838},
  year      = {2025},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  doi       = {10.1109/ACCESS.2025.3592104},
  url       = {http://dx.doi.org/10.1109/ACCESS.2025.3592104}
}
```
Note: Our implementation follows the pipeline provided in https://github.com/IuliaDuta/sheaf_HNN. 
We thank the original authors for releasing their code.

