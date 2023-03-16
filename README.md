# FlowAutoML

How to use it?

The model is divided into two parts: 1. Training; 2. Embedding.

We firstly run `train.py` to obtain `model.h5` and log of `features`. 

When you finish training, you can change the parameter `--isInitialization` to `no`. Next, the saved model can be converted into `.c` and `.h` to realize embedded development.
`MODEL_FILE` is saved model file path. `Test_Example` is data set. `Feature_List`is features log path.