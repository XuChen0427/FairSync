# Submission FairSync

## Prerequisites

- Python 3
- TensorFlow-GPU >= 1.8 (< 2.0)
- pytorch
- Faiss-GPU
- The project is based on ComiRec github

## Getting Started

### Installation

- Install TensorFlow-GPU 1.x

- Install Faiss-GPU based on the instructions here: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md


### Dataset

- Original links of datasets are:

  - http://jmcauley.ucsd.edu/data/amazon/index.html
  - https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1

To fit the requirement of uploading, we only save the processed category files, for the processed dataset, please see ComiRec(KDD2020) https://github.com/THUDM/ComiRec

### Training

#### Firstly, Training on the base retrieval model based on the following command

You can use `python src/train.py --dataset {dataset_name} --model_type {model_name}` to train a specific model on a dataset. Other hyperparameters can be found in the code. (If you share the server with others or you want to use the specific GPU(s), you may need to set `CUDA_VISIBLE_DEVICES`.) 

For example, you can use `python src/train.py --dataset book --model_type ComiRec-DR` to train ComiRec-DR model on Book dataset.



#### Secondly, Process FairSync:

please run `python src/FairSync.py`

demo result:
{'recall': 0.074, 'ndcg': 0.062, 'hitrate': 0.160, 'diversity': 0.208, 'ESP': 1.0}

For other parameters see:

| args_name  | type  | description                                                                        |
|---------|-------|------------------------------------------------------------------------------------|
| dataset | str   | Choose dataset in ["book", "taobao"]                                               |
| model_type | str   | The base model type, please make sure the model is trained in the first step       |
| minimum_exposure | float | Each groups' required minimum exposures                                            |
| topN | int   | Retreival number,  please make sure the model is trained in the first step         |
| FairSync_lr | float | The dual updating learning rate                                                    |
| eval_batch_size | int   | The performing batch size B in the paper                                           |

