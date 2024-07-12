import os
import datasets
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    RagRetriever,
    RagTokenizer,
    RagSequenceForGeneration,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    BartTokenizer
)

from eval_f1_score import f1_score

# OpenMP 오류를 해결하기 위한 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 임베딩 계산 함수
def compute_embeddings(batch, ctx_tokenizer, ctx_encoder):
    inputs = ctx_tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return {'embeddings': embeddings.cpu().numpy().tolist()}

# evaluation model
def evaluate_rag_model(questions, model, tokenizer, retriever):
    model.eval()
    answers = []
    for question in questions:
        # 질문을 토큰화하고 입력 데이터 준비
        inputs = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt")
        input_ids = inputs["input_ids"]
        question_hidden_states = model.question_encoder(**inputs)[0]

        # 검색된 문서 확인
        docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
        
        # 모델을 사용하여 답변 생성
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                context_input_ids=docs_dict["context_input_ids"], 
                doc_scores=doc_scores,
                num_beams=5
                )
            
        # 생성된 토큰을 문자열로 변환
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.append(generated_text[0])
    return answers

def main():
    # 데이터셋 로드
    dataset = datasets.load_dataset('csv', data_files='examples/research_projects/rag/output/psgs_w100.tsv', delimiter='\t')

    # 데이터셋의 첫 번째 1000개 샘플만 사용
    sampled_dataset = dataset['train'].select(range(1000))

    # 문서 임베딩 계산을 위한 사전 훈련된 DPR 모델 로드
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # 임베딩 계산
    sampled_dataset = sampled_dataset.map(lambda batch: compute_embeddings(batch, ctx_tokenizer, ctx_encoder), batched=True, batch_size=8)

    # FAISS 인덱스 추가
    sampled_dataset.add_faiss_index(column='embeddings')

    # Question encoder와 Generator Tokenizer 로드
    question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # RagTokenizer 초기화
    tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)

    # RAG retriever 설정
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        indexed_dataset=sampled_dataset
    )

    # RAG 모델 로드
    model = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq",
        retriever=retriever
    )

    # 데이터셋 준비
    questions = [
        "Where was Anatoly Karpov born?",
        "What is Allen Ginsberg best known for?",
        "What did the term alcohol originally refer to?"
    ]

    gold_answers = [
        "Anatoly Karpov was born on May 23, 1951, at Zlatoust in the Urals region of the former Soviet Union.",
        "Allen Ginsberg is best known for his poem 'Howl', in which he denounced the destructive forces of capitalism and conformity in the United States.",
        "The term alcohol originally referred to the primary alcohol ethanol (ethyl alcohol)."
    ]

    # 모델 평가 실행
    answers = evaluate_rag_model(questions, model, tokenizer, retriever)
    print(answers)
    for question, answer in zip(questions, answers):
        print(f"Question: {question}\nAnswer: {answer}\n")

    # Cosine similarity 계산
    f1 = f1_score(answers, gold_answers)

    print(f"f1 Score: {f1:.4f}")
  

if __name__ == '__main__':
    main()
