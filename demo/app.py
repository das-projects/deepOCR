# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

import matplotlib.pyplot as plt
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import torch

from deepocr.io import DocumentFile
from deepocr.models import ocr_predictor
from deepocr.utils.visualization import visualize_page

DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "master", "sar_resnet31"]


def main():
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("deepOCR: Deep Optical Character Recognition")
    st.write('\n')

    # File selection
    container1 = st.container()
    container1.subheader("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = container1.file_uploader("Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])
    doc = []
    page_idx = 1
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read()).as_images()
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = container1.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
    st.write('\n')

    container2 = st.container()
    container2.subheader("Model Selection")
    det_arch = container2.selectbox("Text detection model", DET_ARCHS)
    st.write('\n')
    reco_arch = container2.selectbox("Text recognition model", RECO_ARCHS)
    st.write('\n')

    if container2.button("Analyze page"):

        cols_image = st.columns((1, 1, 1, 1))
        cols_image[0].write("Input page")
        cols_image[1].write("Segmentation heatmap")
        cols_image[2].write("OCR output")
        cols_image[3].write("Page reconstitution")

        if uploaded_file is not None:
            cols_image[0].image(doc[page_idx])

        if uploaded_file is None:
            container1.subheader("Please upload a document")
        else:
            with st.spinner('Loading model...'):
                predictor = ocr_predictor(det_arch, reco_arch, pretrained=True)

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.det_predictor.pre_processor([doc[page_idx]])
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = torch.squeeze(seg_map[0, ...])
                seg_map = cv2.resize(seg_map.detach().numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols_image[1].pyplot(fig)

                # Plot OCR output
                out = predictor([doc[page_idx]])
                fig = visualize_page(out.pages[0].export(), doc[page_idx], interactive=False)
                cols_image[2].pyplot(fig)

                # Page reconsitution under input page
                # page_export = out.pages[0].export()
                img = out.pages[0].synthesize()
                cols_image[3].image(img, clamp=True)

                # Display JSON
                # st.markdown("\nHere are your analysis results in JSON format:")
                # st.json(page_export)


if __name__ == '__main__':
    main()
