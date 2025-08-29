# 1. 'slim' 버전을 사용하여 기본 이미지 크기를 줄입니다.
FROM python:3.9-slim

# 2. 작업 디렉터리 설정
WORKDIR /app

# 3. pip를 최신 버전으로 업그레이드
RUN pip install --upgrade pip

# 4. 라이브러리 목록 파일을 복사하고 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 모든 소스 코드와 모델 파일을 복사합니다.
COPY . .

# 6. 8000번 포트를 외부에 노출
EXPOSE 8000

# 7. 컨테이너가 시작될 때 uvicorn 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]