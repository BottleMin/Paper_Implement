# 원본 파일 읽기
import pandas as pd

# 1. CSV 파일 읽기
file_path = '--INPUT file--'

# 파일을 라인 단위로 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 2. 라인별로 질문과 답변 분리
processed_lines = []
for line in lines:
    question, answer = line.strip().split('\t', 1)
    processed_lines.append(question)
    processed_lines.append(answer)

# 3. 질문과 답변을 데이터프레임으로 변환
questions = processed_lines[0::2]
answers = processed_lines[1::2]

df = pd.DataFrame({'Question': questions, 'Answer': answers})

# 4. 데이터프레임을 CSV 파일로 저장
csv_file_path = '--OUTPUT file--'
df.to_csv(csv_file_path, index=False)

# 결과 출력 (옵션)
print("CSV 파일이 성공적으로 저장되었습니다:", csv_file_path)
