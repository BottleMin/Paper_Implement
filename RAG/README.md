# Open Domain Question Answering (ODQA)

## Definition

Open Domain Question Answering (ODQA) refers to the task of answering a question by retrieving relevant passages from a Knowledge Base (KB) and generating an answer through a reader model. This task involves searching external data, rather than relying solely on a pre-existing dataset. For example, when asked "When was Sogang University founded?", ODQA does not search within its training data. Instead, it locates a document from an external source like Wikipedia that states, "Sogang University was founded on April 18, 1960, by the Jesuits." The model then extracts the appropriate answer, "April 18, 1960."

## Process of ODQA

### Data and Models

- **Question**: Poses queries that are difficult to answer from internal data alone.
- **Answer**: Assumes there is a span of consecutive tokens within the retrieved document that can be provided as an answer.
- **Knowledge Base (KB)**: An external database comprising millions of documents.
- **Passage**: A document selected from the KB that is relevant to the question.

### Models

- **Retriever**: Finds relevant passages from the KB that relate to the question. It embeds both the question and passages, then computes the inner product to select the top-K passages.
- **Reader**: Extracts potential answer spans from the retrieved passages using a model like BERT.
- **Generator**: Used in RAG to generate answers by combining the question and retrieved passages. BART is typically used for this purpose.

### Workflow
![image](https://github.com/user-attachments/assets/0df1dfd7-9111-4a0a-8c0a-454929cc61a8)
1. The question is input into the system, and both the question and KB are embedded.
2. The inner product between the embedded question and KB is computed to score the passages.
3. The top-K passages are selected based on the scores.
4. The selected passages and question are fed into the reader, which generates hidden representations and extracts the final answer.

## Role of ODQA

### Closed Book Question Answering (CBQA)

- **Definition**: Generates answers without providing external information, relying solely on the knowledge encoded in the model's parameters (Parametric Implicit Knowledge).
- **Examples**: T5 and GPT-3.
- **Advantages**: No need for external data access; end-to-end usability.
- **Disadvantages**: Lack of user control over answer generation; potential for hallucinations.

### Open Domain Question Answering (ODQA)

- **Definition**: Generates answers by incorporating external information, measuring the similarity between the query and passages (Non-Parametric Knowledge).
- **Advantages**: Can provide accurate answers by referencing external sources; easy to update and modify knowledge.
- **Disadvantages**: Limited by the information available in the external sources.

ODQA is significant for improving answer accuracy for questions not covered by the training data. It reduces the need for extensive retraining and leverages external data to overcome the limitations of a fixed training set.

# Retrieval-Augmented Generation (RAG)

## Knowledge Intensive Task (KIT)

Knowledge Intensive Task (KIT) involves generating answers by referencing external knowledge. Unlike ODQA, which finds and extracts spans from passages, KIT generates answers based on the retrieved spans. For example, in response to the question "Is Tokyo the capital of Korea?", ODQA might find a passage stating "The capital of Korea is Seoul." KIT would then generate the answer "No" based on this information.

## RAG Model
![image](https://github.com/user-attachments/assets/bff64185-b944-439f-8ecc-04acf7012720)

Introduced by Facebook AI Research in 2020, the RAG model combines ODQA and KIT characteristics. RAG replaces the traditional reader model with a generator, allowing it to generate answers based on the combination of input passages and questions. This dual approach enables it to handle a wider range of tasks, including those involving questions not directly answerable by existing data.

### Detailed Structure

- **Likelihood Probability**: The likelihood probability of generating an answer given a question, Paq, is maximized using a latent passage variable p.
- **Equation**: $maxPaq=maxPapq=maxPapq×Ppq$
  - **Reader**: Handles Papq.
  - **Retriever**: Handles Ppq.

RAG uses a generator (BART) for answer generation and Dense Passage Retrieval (DPR) for retrieval. The model has two main variants:

### RAG-Sequence Model

- **Description**: Considers the entire sequence when predicting the next token, referring to the retrieved passages.
- **Advantages**: Simpler and computationally efficient, maintaining consistency by sharing the same context across all tokens.

### RAG-Token Model

- **Description**: Considers each token independently when predicting the next token, referring to the retrieved passages.
- **Advantages**: More flexible, allowing richer and more accurate answers by gathering information from various documents.
- **Disadvantages**: More computationally intensive and complex.

### Retriever: DPR

- **Equation**: pηzx∝expdz⊤qx dz=BERTdz qx=BERTqx
- **Description**: Uses dense vector representations for embeddings, providing compact information about documents.
- **Advantages**: Offers more accurate answers by leveraging dense vector space instead of traditional sparse vector models like TF-IDF and BM25.

RAG combines non-parametric and parametric approaches, using BART for generating answers and DPR for effective retrieval, allowing it to address a broader range of questions, including those not covered by existing data.

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

## Simulation Result

| Question                                         | Predicted Answer                                                                                                                               | Gold Answer                                                     |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| When was the peak to peak gondola built          | executive Europe send actions Europe soon Europe asking send actions Europe soon Europe asking send actions Europe soon Europe send actions... | in 2007 and 2008                                                |
| Who starred in an officer and a gentleman        | executive Europe send actions Europe soon Europe asking send actions Europe soon Europe asking send actions Europe soon Europe send actions... | Louis Gossett Jr., Debra Winger, Richard Gere, Phillip J. Salmon, David Keith, Louis Gossett Jr. |
| Who starred in the movie on golden pond          | executive Europe send actions Europe soon Europe asking send actions Europe soon Europe asking send actions Europe soon Europe send actions... | William Lanteau, Christopher Rydell, Katharine Hepburn, Jane Fonda, Doug McKeon, Henry Fonda, Dabney Coleman |

- The predicted answers are nonsensical and incorrect.

F1 score: 0

## Why Did the Prediction Performance Suffer?

- Excessive memory savings during finetuning.
- Performance limitations of the Colab environment.
  - Limited training performance due to sampling 10,000 documents from the .tsv file.

We plan to conduct additional experiments to improve the performance of the RAG model.
