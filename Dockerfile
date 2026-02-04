FROM python:3.10-slim

RUN apt-get update; apt-get -y install curl
RUN apt-get install -y gcc g++ libopenblas-dev
RUN pip install --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-binary scikit-learn scikit-learn

# Add Jupyter Notebook dependencies
RUN pip3 install jupyter notebook

COPY . .