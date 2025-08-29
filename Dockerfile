# 1. Python 3.9 버전을 기반으로 이미지 생성
FROM python:3.9

# 2. 작업 디렉터리 설정
WORKDIR /app

# 3. 라이브러리 목록 파일을 먼저 복사하고 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 모든 소스 코드 복사
COPY . .

# 5. 8000번 포트를 외부에 노출
EXPOSE 8000

# 6. 컨테이너가 시작될 때 uvicorn 서버 실행 (Koyeb PORT 환경 변수 사용)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
