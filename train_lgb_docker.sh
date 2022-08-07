docker build . -t foursquare
docker run --runtime nvidia --shm-size 2G -v /kaggle:/kaggle -e \
  PYTHONUNBUFFERED=1 foursquare train_lgb.sh
