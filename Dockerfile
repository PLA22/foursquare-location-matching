FROM gcr.io/kaggle-gpu-images/python:v116

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_THREAD_MODE=gpu_private

WORKDIR /usr/ml

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train_arcmargin.sh .
COPY train_lgb.sh .
COPY validate_arcmargin.sh .

COPY foursquare foursquare
COPY config config

ENTRYPOINT ["bash"]
CMD []

# docker build . -t foursquare
# docker run -it foursquare /bin/bash
# docker run --runtime nvidia --shm-size 2G -v /kaggle:/kaggle foursquare
