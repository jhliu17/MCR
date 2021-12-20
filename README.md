# MCR: Multi-perspective Coherent Reasoning for Helpfulness Prediction of Multimodal Reviews

Code for the paper "[Multi-perspective Coherent Reasoning for Helpfulness Prediction of Multimodal Reviews](https://aclanthology.org/2021.acl-long.461/)" (ACL 2021).

If you use this code, please cite the paper using the BibTeX reference below.

```bibtex
@inproceedings{mcr,
    title={Multi-perspective Coherent Reasoning for Helpfulness Prediction of Multimodal Reviews},
    author={Junhao Liu, Zhen Hai, Min Yang, and Lidong Bing},
    booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, {ACL} 2021},
    year={2021},
}
```


## Requirements

- Python >= 3.7
- PyTorch >= 1.6

You can install all required Python packages with
```bash
pip install -r requirements.txt
```


## Datasets

To obtain the multimodal datasets of Lazada-MRHP and Amazon-MRHP, please read the details provided [here](scripts/README.md).


## Running the code

### Train
Use the following commands to train the model based on the specific configuration file.
```bash
# single gpu or data parallel, [ckpt] is optional for continual training
sh scripts/train.sh device_ids config_file [ckpt]

# or distributed training
sh scripts/train_dist.sh device_ids n_procs config_file [ckpt]
```

Sample configuration files are provided in the `config` folder.

### Evaluate
Do evaluation on a specific dataset based on the saved model checkpoint and corresponding configuration file.
```bash
sh scripts/eval.sh device_ids config_file ckpt
```


## Contact

This repo is now under active development, and there may be issues caused by refactoring code. If you have any questions, please feel free to email me at junhaoliu17@outlook.com.


## Acknowledgments

Our code is built based on the text matching library [MatchZoo](https://github.com/NTMC-Community/MatchZoo-py) and the PyTorch version [bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch).
