FROM python:3.11.9


COPY ./requirements.txt ./
COPY . /app/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /app

EXPOSE 5000

# ENTRYPOINT ["python",  "app/app.py"]
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
