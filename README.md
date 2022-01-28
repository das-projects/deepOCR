<div align="center">    
 
# DeepOCR     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
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
This is an Optical Character Recognition Library with the ability to train and deploy Deep Neural Network models 
to a Streamlit Web application. The base library in written using PyTorch and PyTorch-Lightning, while the dashboard was 
developed using the Streamlit library.  

## How to run   
First, install dependencies   
```bash
# clone deepOCR   
git clone https://github.com/das-projects/deepOCR

# install deepOCR   
cd deepOCR 
pip install -e .  
sudo apt-get install fonts-freefont-ttf -y 
 ```   
## Streamlit Webapp
 Next, try out the streamlit dashboard for a demonstration   
 ```bash
# demo folder
cd deepOCR
# run demo
streamlit run demo/app.py    
```

## Python code API
This project is setup as a package which means you can now easily import any file into any other file:
```bash
# Download an example image
wget https://eforms.com/download/2019/01/Cash-Payment-Receipt-Template.pdf
```
```python
import matplotlib.pyplot as plt

from deepocr.io import DocumentFile
from deepocr.models import ocr_predictor

# Load the pdf file
doc = DocumentFile.from_pdf("Cash-Payment-Receipt-Template.pdf").as_images()
print(f"Number of pages: {len(doc)}")

# Use the predictor object to detect and recognize text
predictor = ocr_predictor(pretrained=True)

# show the predictor output!
result = predictor(doc)
result.show(doc)

# Use synthesize method to regenerate the image in a desired format 
synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()
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
