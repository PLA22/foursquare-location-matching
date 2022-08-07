python3 foursquare/pipeline_train_arcmargin.py \
  --path-config ./config/config_arcmargin_tfidf.json \
  --dir-experiment /kaggle/models/arcmargin/tfidf_512 \
  --path-locations /kaggle/input/foursquare-location-matching/train_fold.csv \
  --fold 1

python3 foursquare/pipeline_train_arcmargin.py \
  --path-config ./config/config_arcmargin_tfidf.json \
  --dir-experiment /kaggle/models/arcmargin/tfidf_512 \
  --path-locations /kaggle/input/foursquare-location-matching/train_fold.csv \
  --fold 2

python3 foursquare/pipeline_train_arcmargin.py \
  --path-config ./config/config_arcmargin_tfidf.json \
  --dir-experiment /kaggle/models/arcmargin/tfidf_512 \
  --path-locations /kaggle/input/foursquare-location-matching/train_fold.csv \
  --fold 3
