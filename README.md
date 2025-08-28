# Hypergraph Neural Sheaf Diffusion (HNSD)

This repository contains the official implementation of the paper:

**"Hypergraph Neural Sheaf Diffusion: A Symmetric Simplicial Set Framework for Higher-Order Learning"** 

---

## Requirements
- Python >= 3.7
- PyTorch >= 1.7
- scikit-learn
- matplotlib
- numpy

(You may add other dependencies as needed.)

---

## Usage

### Training
```bash
python train.py --dname <DATASET_NAME> --method SheafHyperGNNDiag --new_edge True --earlystop True --plot True
```

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

