DIR_MODELS_ARCMARGIN="/kaggle/models/arcmargin"
DIR_MODELS_LGB="/kaggle/models/lgb"

DIR_SUBMISSION_MODELS="submission/models"
DIR_SUBMISSION_ARCMARGIN="submission/models/arcmargin"
DIR_SUBMISSION_LGB="submission/models/lgb"

SUBMIT_MODELS="0"
EXP1_ARCMARGIN="tfidf_512"
EXP2_ARCMARGIN="xlm"
EXP_LGB="20220707_130025"

# upload code
rm -r submission/src/foursquare/*
cp -r foursquare submission/src
cp config/config.json submission/src
kaggle datasets version -p submission/src -r zip -m "new src version"

# upload pretrained models
if [ $SUBMIT_MODELS = "1" ]
then
  rm -r $DIR_SUBMISSION_ARCMARGIN
  mkdir -p $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN
  cp -r $DIR_MODELS_ARCMARGIN/$EXP1_ARCMARGIN/* $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN/*/model.ckpt*
  rm -r $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN/*/model_emb
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN/*/valid_embeddings.npy
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP1_ARCMARGIN/*/oof_embeddings.npy

  mkdir -p $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN
  cp -r $DIR_MODELS_ARCMARGIN/$EXP2_ARCMARGIN/* $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN/*/model.ckpt*
  rm -r $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN/*/model_emb
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN/*/valid_embeddings.npy
  rm $DIR_SUBMISSION_ARCMARGIN/$EXP2_ARCMARGIN/*/oof_embeddings.

  rm -r $DIR_SUBMISSION_LGB
  mkdir -p $DIR_SUBMISSION_LGB
  cp -r $DIR_MODELS_LGB/$EXP_LGB/* $DIR_SUBMISSION_LGB

  kaggle datasets version -p $DIR_SUBMISSION_MODELS -r zip -m "new models version"

  sleep 600
fi

# create new kaggle dataset
#kaggle datasets init -p $DIR_SUBMISSION_MODELS
#sed -i 's/INSERT_TITLE_HERE/foursquare-models/' $DIR_SUBMISSION_MODELS/dataset-metadata.json
#sed -i 's/INSERT_SLUG_HERE/foursquare-models/' $DIR_SUBMISSION_MODELS/dataset-metadata.json
#kaggle datasets create -p $DIR_SUBMISSION_MODELS -r tar

# upload notebook (warning: maybe the datasetS are still creating and
# could not be added te the notebook)
sleep 10
kaggle kernels push -p submission/nb
