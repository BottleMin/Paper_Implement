# Retrieval-Augmented Generation (RAG)

This project implements and experiments with the Retrieval-Augmented Generation (RAG) model, which combines features of Open Domain Question Answering (ODQA) and Knowledge Intensive Tasks (KIT). RAG enhances the accuracy and reliability of generated answers by integrating ODQA and KIT characteristics.

## Introduction

The RAG model, proposed by the Facebook AI Research team in the paper ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401), combines the strengths of ODQA and KIT to generate more accurate answers. This project is based on the work from the [Hugging Face transformers example repository](https://github.com/huggingface/transformers/tree/3345ae733b6f4aeb7204a0f3e646a3cdbaad0023/examples/research_projects/rag) and the [official Hugging Face documentation](https://huggingface.co/docs/transformers/main/en/model_doc/rag).

## Installation

To set up the project, follow these steps:

Download the necessary data. Use the DPR data download script or directly download the [psgs_w100.tsv.gz file](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).

Additional steps:
- Download the [nq-train.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv) file.
- Download the [nq-test.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv) file.
- Use `process_qa_file.ipynb` to clearly separate the Question and Answer in the .csv files.
- Use `embedding_tsv.ipynb` to generate a FAISS index file from your Wikipedia data.

## Finetuning

Additional finetuning steps:
- The `facebook/rag-sequence-base` model was selected as the pretrained model for `RagSequenceForGeneration`.
- Custom FAISS files will be used for the retriever's document store.
- To optimize GPU memory usage, the following measures were implemented:
  - Enabled Mixed Precision Training with `precision='16-mixed'`.
  - Applied Gradient Accumulation with `accumulate_grad_batches=2`.
  - Deactivated specific layers to reduce memory consumption:

    ```python
    for name, param in self.model.generator.named_parameters():
        parts = name.split('.')
        if 'layers' in name and parts[1].isdigit():
            layer_num = int(parts[1])
            if layer_num > 6:
                param.requires_grad = False
    ```
  - Set the batch size to 2.

## Evaluation

- Conducted end-to-end (e2e) evaluation.
- F1 scores were calculated for the downstream task.

This README file provides a detailed overview of the project setup, finetuning, and evaluation processes. For more information, refer to the respective notebooks and scripts included in the repository.
