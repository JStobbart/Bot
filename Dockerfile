FROM python:3.9

WORKDIR /home
ENV API_TOKEN="YOUR API_TOKEN"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]