FROM python:3.11

ENV CNSTD_DOWNLOAD_SOURCE="HF"
ENV CNOCR_DOWNLOAD_SOURCE="HF"

# Create app directory
WORKDIR /app

# Install app dependencies
COPY ./requirements.txt ./


RUN pip install -r requirements.txt

COPY . .

EXPOSE 50051
CMD [ "python", "server.py" ]
