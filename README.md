<div align="center">    
 
# DeepOCR     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone deepOCR   
git clone https://github.com/das-projects/deepOCR

# install deepOCR   
cd deepOCR 
pip install -e .   
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd deepOCR
# run demo
streamlit run demo/app.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python

from pytorch_lightning import Trainer

# model

# train
trainer = Trainer()


# test using the best model!

```

### Citation   
```
@article{Arijit Das,
  title={deep OCR: Optical Character Recognition with Deep Learning},
  author={Arijit Das},
  journal={https://github.com/das-projects/deepOCR},
  year={2022}
}
```   
