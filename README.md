# Hypergraph Neural Sheaf Diffusion (HNSD)

This repository contains the official implementation of the paper:

**"Hypergraph Neural Sheaf Diffusion: A Symmetric Simplicial Set Framework for Higher-Order Learning"** 

---

## Requirements
Install the dependencies with:
```bash
pip install -r requirements.txt
```
---

## Usage

### Training
```bash
cd src
python train.py --dname senate-committees --method SheafHyperGNNDiag --new_edge True --earlystop True --plot True
```

## Dataset Preparation
Follow the dataset preparation steps from the [AllSet repository](https://github.com/jianhao2016/AllSet).  
then place the resulting `data` folder under `../data`.

## Citation

If you find this code useful in your research, please cite:

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

Note: Our implementation builds upon the pipeline from [sheaf_HNN](https://github.com/IuliaDuta/sheaf_HNN). 
We thank the original authors for releasing their code.

