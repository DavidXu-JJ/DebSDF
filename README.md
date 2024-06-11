
<p align="center">

  <h1 align="center">DebSDF: Delving into the Details and Bias of Neural Indoor Scene Reconstruction</h1>
  <h3 align="center">TPAMI 2024</h3>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=AioOVwEAAAAJ&hl=en"><strong>Yuting Xiao<sup>1*</sup></strong></a>
    ·    
    <a href="https://davidxu-jj.github.io/"><strong>Jingwei Xu<sup>1*</sup></strong></a>
    ·
    <a href="https://niujinshuchong.github.io/"><strong>Zehao Yu<sup>2</sup></strong></a>
    ·
    <a href="https://scholar.google.com.sg/citations?user=fe-1v0MAAAAJ&hl=en"><strong>Shenghua Gao<sup>1†</sup></strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2308.15536"><img src='https://img.shields.io/badge/arXiv-DebSDF-red' alt='Paper PDF'></a>
        <a href="https://davidxu-jj.github.io/pubs/DebSDF/"><img src='https://img.shields.io/badge/Project_Page-DebSDF-green' alt='Project Page'></a>
    <br>
    <b> ShanghaiTech University |&nbsp;University of Tübingen  </b>
    <br>
    <b> (<sup>*</sup> denotes equal contributions, † denotes corresponding author)</b>
    </p>

  <table align="center">
    <tr>
    <td>
      <img src="media/teaser.png">
    </td>
    </tr>
  </table>

## News
* **[2024.6.11]** Code release.
* **[2024.6.6]** Training data released.
* **[2024.6.5]** Accepted to TPAMI 2024. Congratulations to every author.


## Demo

https://github.com/DavidXu-JJ/DebSDF/assets/68705456/09ec3cf1-b17e-4efe-a661-51c85e815804

More results are presented in the [project page](https://davidxu-jj.github.io/pubs/DebSDF/).

## Setup

```
conda create -n DebSDF python=3.10
conda activate DebSDF

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

For the preprocessed training data, please download from [google drive](https://drive.google.com/drive/folders/1bTTIxfaHnX-3bfTt-QIzbsAvtn96BZ53?usp=sharing).

Please extract the data to the `./data` folder.

Our training data is adapted from [MonoSDF](https://github.com/autonomousvision/monosdf). The ICL data please view from original author's [webpage](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).

Some scene files have been provided by various artists for free on BlendSwap. Please refer to the table below for license information and links to the .blend files. (denoted as neural_rgbd_data in google drive)

| License       | Scene name                                             |
| ------------- | ------------------------------------------------------ |
| CC-BY         | [Breakfast room](https://blendswap.com/blend/13363)    |
| CC-0          | [Complete kitchen](https://blendswap.com/blend/11801)  |
| CC-BY         | [Green room](https://blendswap.com/blend/8381)         |
| CC-BY         | [Grey-white room](https://blendswap.com/blend/13552)   |
| CC-BY         | [Kitchen](https://blendswap.com/blend/5156)            |
| CC-0          | [Morning apartment](https://blendswap.com/blend/10350) |
| CC-BY         | [Whiteroom](https://blendswap.com/blend/5014)          |

### Run With Your Own Data

Please consider following the guidelines provided by [MonoSDF](https://github.com/autonomousvision/monosdf?tab=readme-ov-file#high-resolution-cues).

## Code

### Scannet (scan_id from 1 to 4) running example:

```
cd code
python -m torch.distributed.launch  --master_port 22112 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf ./confs/scannet_mlp.conf  --scan_id $scan_id
```

or you can run:

```
cd code
sh train.sh ./confs/scannet_mlp.conf 1 0
(means run the scan 1 of scannet with GPU 0)
```

### BlenderSwap (scan_id from 1 to 7) running example:

```
cd code
python -m torch.distributed.launch  --master_port 22112 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf ./confs/neural_rgbd_mlp.conf  --scan_id $scan_id
```

or you can run:

```
cd code
sh train.sh ./confs/neural_rgbd_mlp.conf 1 0
(means run the scan 1 of selected blender scenes with GPU 0)
```

### Replica (scan_id from 1 to 8) running example:

```
cd code
python -m torch.distributed.launch  --master_port 22112 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf ./confs/replica_mlp.conf  --scan_id $scan_id
```

or you can run:

```
cd code
sh train.sh ./confs/replica_mlp.conf 1 0
(means run the scan 1 of replica with GPU 0)
```

### TNT Advanced(scan_id from 1 to 4) running example:

Please make sure the config file corresponds to the scan id.

```
cd code
python -m torch.distributed.launch  --master_port 22112 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf ./confs/tnt_mlp_1.conf  --scan_id 1
```

or you can run:

```
cd code
sh train.sh ./confs/tnt_mlp_1.conf 1 0
(means run the scan 1 of tnt advanced with GPU 0)

sh train.sh ./confs/tnt_mlp_2.conf 2 1
(means run the scan 2 of tnt advanced with GPU 1)
```


## Evaluation

### Additional Setup

```
pip install -r evaluation_requirements.txt
```

### Evaluate Scannet

```
cd scannet_eval
python3 evaluate.py
```

### Evaluate BlenderSwap by artists (denoted as neural rgbd for convience)

```
cd neural_rgbd_eval
python3 evaluate.py
```

### Evaluate Replica

```
cd replica_eval
python3 evaluate.py
```

### Evaluate TNT Advanced

You need to submit the reconstruction results to the [official evaluation server](https://www.tanksandtemples.org), please follow their guidance. We provide an example of our submission at [this link](https://drive.google.com/drive/folders/1gAgrNOkKPlOb1PdbAEqig2WMnCK--zh7?usp=sharing).

## Acknowledgement

Our code is based upon [MonoSDF](https://github.com/autonomousvision/monosdf). Additional training data is also processed with [Omnidata](https://omnidata.vision/) for monocular depth and normal extraction.

## Bibtex
```bibtex
@misc{xiao2023debsdf,
      title={DebSDF: Delving into the Details and Bias of Neural Indoor Scene Reconstruction}, 
      author={Yuting Xiao and Jingwei Xu and Zehao Yu and Shenghua Gao},
      year={2023},
      eprint={2308.15536},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
