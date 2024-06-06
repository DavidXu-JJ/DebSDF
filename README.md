
<p align="center">

  <h1 align="center">DebSDF: Delving into the Details and Bias of Neural Indoor Scene Reconstruction</h1>
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
* **[On-going]** Organizing the code for release.
* **[Finished]** Training data released.
* **[2024.6]** Accepted to TPAMI 2024. Congratulations to every author.


## Video Preview

https://github.com/DavidXu-JJ/DebSDF/assets/68705456/09ec3cf1-b17e-4efe-a661-51c85e815804

More results are presented in the [project page](https://davidxu-jj.github.io/pubs/DebSDF/).



## Dataset

For the preprocessed training data, please download from [google drive](https://drive.google.com/drive/folders/1bTTIxfaHnX-3bfTt-QIzbsAvtn96BZ53?usp=sharing).

Our training data is adapted from [MonoSDF](https://github.com/autonomousvision/monosdf).

Some scene files have been provided by various artists for free on BlendSwap. Please refer to the table below for license information and links to the .blend files.

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

Working in process.

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
