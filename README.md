## Confusionset-guided Pointer Networks for Chinese Spelling Check (ACL2019)

## Main Libraries
* Python 37
* Pytorch 0.4
* numpy
## Train a model for Chinese Spelling Check
```
>>> python train.py -pes=  -trp= -vap=your -tsp=

```
## Reference Resources
---

SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

SIGHAN Bake-off 2014 : http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

SIGHAN Bake-off 2015 : http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

**Note**: All datasets above are originally written in Traditional Chinese. Considering the fact that our generated  datasets are in Simplified Chinese, we have translated
the original datasets into a version of Simplified Chinese, which can be found in the **data/sighan** folder. The tool we use to translate Tranditional Chinese to Simplified
Chinese is [OpenCC](https://github.com/BYVoid/OpenCC).

Automatic-Corpus-Generation: https://github.com/wdimmy/Automatic-Corpus-Generation

## Citation

If you find the implementation useful, please cite the following paper:
**Confusionset-guided pointer networks for Chinese spelling check**
```buildoutcfg
@inproceedings{wang2019confusionset,
  title={Confusionset-guided pointer networks for Chinese spelling check},
  author={Wang, Dingmin and Tay, Yi and Zhong, Li},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={5780--5785},
  year={2019}
}
```

## Contact
Drop me (Dingmin Wang) an email at wangdimmy (AT) gmail.com if you have any question.







