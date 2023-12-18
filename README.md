# PLSPL
code for Personalized long-and short-term preference learning for next POI recommendation. TKDE'20

# Important Packages:
- Python
- Pytorch

# Data preprocess
The sequence is preprocessed by a sliding window. The length of each session is 20. The last one is the ground truth. You can change the setting based on your model.

Run data preprocess:

`python preprocess_longshort.py`

In the preprocess.py, I also add the preprocessing for other compared baselines. You can omit them if you don't use them.

# Run

`python train_long_short.py`


If you use the code, please cite the following paper:

```
@article{wu2020personalized,
  title={Personalized long-and short-term preference learning for next POI recommendation},
  author={Wu, Yuxia and Li, Ke and Zhao, Guoshuai and Xueming, QIAN},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2020},
  publisher={IEEE}
}
```
