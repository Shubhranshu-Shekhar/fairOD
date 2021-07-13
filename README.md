# FairOD: Fairness-aware Outlier Detection

FairOD is a fairness-aware outlier detector that has the following desirable properties: (1) exhibits treatment parity at test time, (2) aims to flag equal proportions of samples from all groups (i.e. obtain group fairness, via statistical parity), and (3) strives to flag truly high-risk samples within each group. This repository provides a python implementation of FairOD.

## Resources
* Paper: [FairOD](paper/fairOD-aies-21.pdf)
* Foils: [Deck](slides/fairod-slides.pdf)
* Supplementary: [Appendix](appendix/appendix.pdf)


#### Training using your own dataset
1. Create data directory containing three files: X (feature representation), y(ground-truth lables), and S (protected attribute values)
2. See fairod/run.py for demo config 


#### Installation
You can install the trainer module using the following.
```
pip install git+https://github.com/Shubhranshu-Shekhar/fairOD.git
``` 


## Cite this work:
If you find our work useful, please cite our work:

```
@inproceedings{shekhar2021fairod,
  title={FairOD: Fairness-aware Outlier Detection},
  author={Shekhar, Shubhranshu and Shah, Neil and Akoglu, Leman},
  booktitle={Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  year={2021}
}
```