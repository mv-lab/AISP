# RISP
### Reverse ISP for RGB to Raw mapping

## Setup

Install the dependencies with pip

```
pip install -r requirements.txt

```

Download the candidate models

* [s7](https://drive.google.com/file/d/16PRMmzgzMSkZmK0IzzlA6qcKQl8ZFqjj/view?usp=sharing)
* [p20](https://drive.google.com/file/d/1WvrV0HWtQnA_SrBZ1FcS_GhvC3bVReHy/view?usp=sharing)

Place them into ``checkpoints/`` folder.

To run inference, modify ``launch-inference.sh`` accordingly and run:

```
sh launch-inference.sh
```
