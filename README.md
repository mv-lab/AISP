# AI Image Signal Processing

[Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)

Computer Vision Lab, CAIDAS, University of Würzburg

#### Contents

1. [AIM 2022 Reversed ISP Challenge](#aim-2022-reversed-isp-challenge) Official repo!
1. [Model-Based Image Signal Processors via Learnable Dictionaries](https://ojs.aaai.org/index.php/AAAI/article/view/19926) AAAI 2022 Oral - Official repo!
1. [MAI 2022 Learned ISP Challenge](#mai-2022-learned-isp-challenge) Complete Baseline solution
1. [Citation and Acknowledgement](#citation-and-acknowledgement)
1. [Contact](#contact)

---------------------------------------------------

## [AIM 2022 Reversed ISP Challenge](https://data.vision.ee.ethz.ch/cvl/aim22/) 

### [Track 1 - S7](https://codalab.lisn.upsaclay.fr/competitions/5079) | [Track 2 - P20](https://codalab.lisn.upsaclay.fr/competitions/5080)

<a href="https://data.vision.ee.ethz.ch/cvl/aim22/"><img src="https://i.ibb.co/VJ7SSQj/aim-challenge-teaser.png" alt="aim-challenge-teaser" width="400" border="0"></a>

In this challenge, we look for solutions to recover RAW readings from the camera using only the corresponding RGB images processed by the in-camera ISP. Successful solutions should generate plausible RAW images, and by doing this, other downstream tasks like Denoising, Super-resolution or Colour Constancy can benefit from such synthetic data generation. Click [here to read more information](aim22-reverseisp/README.md) about the challenge.

### Starter guide and code 🔥

- **[aim-starter-code.ipynb](aim22-reverseisp/aim-starter-code.ipynb)** - Simple dataloading and visualization of RGB-RAW pairs + other utils.
- **[aim-baseline.ipynb](aim22-reverseisp/aim-baseline.ipynb)** - End-to-end guide to load the data, train a simple UNet model and make your first submission!


------

## [Model-Based Image Signal Processors via Learnable Dictionaries](https://ojs.aaai.org/index.php/AAAI/article/view/19926) (AAAI '22 Oral)

[Project website](https://mv-lab.github.io/model-isp22/) where you can find the poster, presentation and more information.

> Hybrid model-based and data-driven approach for modelling ISPs using learnable dictionaries. We explore RAW image reconstruction and improve downstream tasks like RAW Image Denoising via raw data augmentation-synthesis.

The code will be released soon. If you have implementation questions or you need qualitative samples for comparison, please contact me.

We provide the figure/illustration of our method in [mbispld](mbispld/mbispld.pdf).

------


## [MAI 2022 Learned ISP Challenge](https://codalab.lisn.upsaclay.fr/competitions/1759)

You can find at [mai22-learnedisp](mai22-learnedisp/) and end-to-end baseline: dataloading, training top solution, model conversion to tflite.
The model achieved 23.46dB PSNR after training for a few hours. Here you can see a sample RAW input and the resultant RGB.

<img src="mai22-learnedisp/result-isp3.png" width="400" border="0">

We test the model on AI Benchmark. The model average latency is 60ms using a input RAW image `544,960,4` and generating a RGB `1088,1920,3`, in a mid-level smartphone (45.4 AI-score) using Delegate GPU and FP16.


------

## Related Work

[1] [Model-Based Image Signal Processors via Learnable Dictionaries](https://ojs.aaai.org/index.php/AAAI/article/view/19926) by Conde et al, AAAI 2022.

[2] [Learned Smartphone ISP on Mobile NPUs with Deep Learning, Mobile AI 2021 Challenge: Report](https://arxiv.org/abs/2105.07809) by Ignatov et al, CVPRW 2021.

[3] [Learning Raw Image Reconstruction-Aware Deep Image Compressors](https://abhijithpunnappurath.github.io/pami_raw.pdf) by Abhijith Punnappurath and Michael S. Brown, TPAMI 2019.

[4] [Unprocessing Images for Learned Raw Denoising](https://arxiv.org/abs/1811.11127) by Brooks et al. , CVPR 2019

[5] [CycleISP: Real Image Restoration via Improved Data Synthesis](https://arxiv.org/abs/2003.07761) by Zamir et al. , CVPR 2020

**Related Challenges**

[Mobile AI & AIM 2022 Learned Smartphone ISP Challenge](https://codalab.lisn.upsaclay.fr/competitions/1759) organized by Andrey Ignatov.


-----------------

## Citation and Acknowledgement

```
@inproceedings{conde2022model,
  title={Model-Based Image Signal Processors via Learnable Dictionaries},
  author={Conde, Marcos V and McDonagh, Steven and Maggioni, Matteo and Leonardis, Ales and P{\'e}rez-Pellitero, Eduardo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={481--489},
  year={2022}
}

@inproceedings{conde2022aim,
  title={{R}eversed {I}mage {S}ignal {P}rocessing and {RAW} {R}econstruction. {AIM} 2022 {C}hallenge {R}eport},
  author={Conde, Marcos V and Timofte, Radu and others},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
  year={2022}
}
```


## Contact

Marcos Conde (marcos.conde-osorio@uni-wuerzburg.de) and Radu Timofte (radu.timofte@uni-wuerzburg.de) are the contact persons and direct managers of the AIM challenge. Please add in the email subject "AIM22 Reverse ISP Challenge" or "AISP"
