# GroundingSAM-for-OpenPIV
## Introduction
This library stores the integrated workflow of GroundingDINO, Segment Anything, and Open-PIV-python-cpu. The use of GroundingDINO and Segment Anything uses the workflow proposed in the library _https://github.com/AliRKhojasteh/Flow_segmentation_ and _https://github.com/idea-research/grounded-segment-anything_ and is modified to remove dependencies on source codes pulled from github. This modified version relies only on PyPI to facilitate easier setup.

## Installation
To install and run the repository, make sure CUDA is installed and configured correctly.

Upon doing so, in the desired virtual environment run:
```console
pip install -r 'requirements.txt'
```

after doing so, run

```console
pip install git+https://github.com/ali-sh-96/openpiv-python-cpu
```

to install oepnpiv-python-cpu

## Notebook Description
There are three notebooks for this repository:

GroundedSAMClean: Step by step workflow of the automated script in the Scripts folder

AutoGS: Simplified mask generation to test the script in Scripts folder

OpenPIVwGS: Integrated workflow of PIV analysis with the automated mask generation
