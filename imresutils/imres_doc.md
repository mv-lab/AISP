# RAW Image Super-Resolution and Restoration Utils
### https://huggingface.co/datasets/marcosv/rawir

You can use these functions to add noise, blur, and downsample RAW images.
See the demo in `generate_lq.ipynb`. You can find datasets at: https://huggingface.co/datasets/marcosv/rawir , and more tips and samples!

This code is designed for the NTIRE Challenges:
- [NTIRE 2024 RAW Image Super Resolution Challenge](https://codalab.lisn.upsaclay.fr/competitions/17631)
- [NTIRE 2025 RAW Restoration Challenge: Track1) Super-Resolution](https://codalab.lisn.upsaclay.fr/competitions/21644)
- [NTIRE 2025 RAW Restoration Challenge: Track2) Restoration](https://codalab.lisn.upsaclay.fr/competitions/21647)

This code was used in our papers:
- [BSRAW: Improving Blind RAW Image Super-Resolution](https://arxiv.org/abs/2312.15487), WACV 2024
- [Deep RAW Image Super-Resolution. A NTIRE 2024 Challenge Survey](https://arxiv.org/abs/2404.16223), CVPRW 2024
- [Toward Efficient Deep Blind Raw Image Restoration](https://arxiv.org/abs/2409.18204), ICIP 2024

--------

## The RAWIR Dataset
### https://huggingface.co/datasets/marcosv/rawir

This dataset includes images different smartphones: iPhoneX, SamsungS9, Samsung21, Google Pixel 7-9, Oppo vivo x90.

**How are the RAW images?**

- All the RAW images in this dataset have been standarized to follow a Bayer Pattern `RGGB`, and already white-black level corrected.
- Each RAW image was split into several crops of size 512x512x4 (1024x1024x3 for the corresponding RGBs). You see the filename `{raw_id}_{patch_number}.npy`.
- For each RAW image, you can find the associated metadata `{raw_id}.pkl`.
- RGB images are the corresponding captures from the phone i.e., the phone imaging pipeline (ISP) output. The images are saved as lossless PNG 8bits.
- Scenes include indoor/outdoor, day/night, different ISO levels, different shutter speed levels.

- How can I load these RAW images?

```
import numpy as np
raw = np.load("raw.npy")
max_val = 2**12 -1
raw = (raw / max_val).astype(np.float32)
```

- How do we save them?

```
import numpy as np
max_val = 2**12 -1
raw = (raw * max_val).astype(np.uint16)
np.save(os.path.join(SAVE_PATH, f"raw.npy"), raw_patch)
```


