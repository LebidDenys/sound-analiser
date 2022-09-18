FROM python:3.9

WORKDIR '/app'

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 ffmpeg
COPY . .

CMD ["python3", "-m", "src.main", "--mode", "predict"]