# Retrieval-Augmented Generation (RAG)
---

This project implements and experiments with the Retrieval-Augmented Generation (RAG) model, which combines features of Open Domain Question Answering (ODQA) and Knowledge Intensive Tasks (KIT). RAG enhances the accuracy and reliability of generated answers by integrating ODQA and KIT characteristics.

## Introduction
The RAG model, proposed by the Facebook AI Research team in the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" ([paper link](https://arxiv.org/abs/2005.11401)), combines the strengths of ODQA and KIT to generate more accurate answers. This project is based on the work from the [Hugging Face transformers example repository](https://github.com/huggingface/transformers/tree/3345ae733b6f4aeb7204a0f3e646a3cdbaad0023/examples/research_projects/rag) and the official [Hugging Face documentation](https://huggingface.co/docs/transformers/main/en/model_doc/rag).

## Installation
To set up the project, follow these steps:

1. Download the necessary data. Use the [DPR data download script](https://github.com/facebookresearch/DPR/tree/main/dpr/data/download_data.py) or directly download the [psgs_w100.tsv.gz file](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

---

This README provides a clear explanation of the project's purpose and installation process. Adjust the details to fit your project's requirements and include additional information as needed. If you need further assistance or have questions, feel free to ask!
