# 1단계: 빌드용 이미지 (Builder)
# 'slim' 버전을 사용하여 기본 이미지 크기를 줄입니다.
FROM python:3.9-slim as builder

WORKDIR /app

# pip를 최신 버전으로 업그레이드
RUN pip install --upgrade pip

# requirements.txt를 먼저 복사하고 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 2단계: 최종 실행용 이미지
FROM python:3.9-slim

WORKDIR /app

# 1단계(builder)에서 설치한 라이브러리만 그대로 복사해옵니다.
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# 나머지 소스 코드와 모델 파일을 복사합니다.
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]