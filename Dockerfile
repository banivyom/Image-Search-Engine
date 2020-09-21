FROM python:3.6

# RUN apk add --no-cache python3-dev py3-pip \
#     && pip3 install --upgrade pip


WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

RUN pip3 install tensorflow
RUN pip3 --no-cache-dir install -r requirements.txt

# EXPOSE 5000


ENTRYPOINT ["python3"]
CMD ["app.py"]