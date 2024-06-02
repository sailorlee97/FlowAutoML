# FlowAutoML

How to use it?

The model is divided into two parts: 1. Training; 2. Embedding.

We firstly run `train.py` to obtain `model.h5` and log of `features`. 

When you finish training, you can change the parameter `--isInitialization` to `no`. Next, the saved model can be converted into `.c` and `.h` to realize embedded development.
`MODEL_FILE` is saved model file path. `Test_Example` is data set. `Feature_List`is features log path.

Citation
If you find this useful in your research, please consider citing:
```
@INPROCEEDINGS{10538990,
  author={Li, Zeyi and Zhang, Ze and Fu, Mengyi and Wang, Pan},
  booktitle={2023 IEEE 22nd International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)}, 
  title={A novel network flow feature scaling method based on cloud-edge collaboration}, 
  year={2023},
  volume={},
  number={},
  pages={1947-1953},
  keywords={Training;Cloud computing;Privacy;Computational modeling;Collaboration;Telecommunication traffic;Stability analysis;Feature Engineering;Feature Scaling;Traffic Classification;Deep Learning;Zero-touch network},
  doi={10.1109/TrustCom60117.2023.00265}}
```
