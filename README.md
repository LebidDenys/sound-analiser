# War analyser

Script to read audio stream by chunks and decode into numpy array (readStream method),
then this array can be used to detect if there was explosion


to run in docker:
```shell
docker build -t image-tag .
docker run image-tag
```

install dependencies:
```shell
python -m pip install requirements.txt
```
run script:
```shell
python src.main --mode [train/predict, default predict] --url [URL(?)|PATH, default='.input/explosions.wav'] --login 'login' --password 'password'
```

## Run script
To run predictions on new data and see results, run 
```shell
python src/main.py --mode predict --stream_url 'your url'
```

The output will be in the `output` folder with `inference_output` and input filename in its name.

### Example on available data and best model so far
```shell
python src/main.py --mode predict --audio_url input/explosions.wav
```

OR (since this is the default URL and default mode)

```shell
python src/main.py
```

The output is then written to the `output/inference_output_model4_explosionswav.csv` file.

## Script modes
1. `'predict'`: default mode. Streams predictions on-the-fly to the endpoint of need.
2. `'train'`: Trains new model on specified data and labels. Iterates through different column configurations and hyperparameter grids.
3. `'evaluate'`: Evaluates model performance given data (and corresponding labels) that's different to the data the model was trained on.

### Predict
Input stream, analyze and output predictions. All on-line

TBD

### Train
Train a new XGBoost model. 

Make sure you have input stream and you have it labeled. The program reads all stream, splits it to training and testing sets and trains the model, searching for best hyperparameter and feature combinations

To label the data, you can use Audino: `https://github.com/midas-research/audino`. The output JSON with labels should be same format as the one in `artifacts/ground truths/exp.json`

### Evaluate
Evaluate trained model on the new input. 

Make sure you have input stream and you have it labeled. The program reads all stream, gives predictions on separate chunks and compares the predictions with the real data.

To label the data, you can use Audino: `https://github.com/midas-research/audino`. The output JSON with labels should be same format as the one in `artifacts/ground truths/exp.json`
## YAML config file
Coordinates should be in decimal degree (decimal notation, DD.DDDDDD°, example: Lat 52.514487, Lng 13.350126) and type float
```yaml
appName: 'Sound analyzer' # application name
prediction_threshold: 0.7 # if probability predicted by system is higher than threshold - sound is considered as explosion/gunshot
explosion_cache_lifetime_ms: 2000 # how long after explosion item would live in cache (i.e. wait for explosions in other points)
endpoints: # TBD
  - url: './input/explosions.wav'
    lat: 'x1'
    lng: 'y1'
  - url: './input/explosions.wav'
    lat: 'x2'
    lng: 'y2'
  - url: './input/explosions.wav'
    lat: 'x3'
    lng: 'y3'
packet_size: 262144 # determines chunk duration = packet_size / (sample_rate * 4)
sample_rate: 44100 # Hz
model_id: 4 # Model ID, list of models by their IDs and their performance is listed in the next section
labels_path: artifacts/ground truths/exp.json # path to the JSON with labeled data that was created using Audino annotator
```

## Models Performance by their ID
For models 1, evaluation performed on test set of `explosions.wav`. For models 2-6, evaluations performed on `ЛЕВЫЙ.wav`.
- model 1: 47.5%
- model 2: 87.6% **but no gunshots classified** (explosions.wav test set - 62.5%)
- model 3: 87.6% **but no gunshots classified** (explosions.wav test set - 60%)
- model 4: 87.6% **but no gunshots classified**
- model 5: 73.53%
- model 6: 97.06%
- model 7: 97.94% - 131072 packet size (0.74 sec duration)
- model 8: 96.76% **but with recall for class 1 being 2.22%** - 32768 packet size (0.18 sec duration)
- model 9: 97.76% **but with recall for class 1 being 14.5%** - 16384 packet size  (0.09 sec duration)
- model 10: 98.09% **but with recall for class 1 being 5.96%** - 4096 packet size (0.02 sec duration)
- model 11: 98.21% **but with recall for class 1 being 4.22%** - 1024 packet size (0.0058 sec duration)
- model 12: 98.24% **but with recall for class 1 being 4.77%** - 512 packet size (0.0029 sec duration)

For more comprehensive model performance stats, refer to the `artifacts/models_results_logs.ods` spreadsheet.

## Other datasets
Google Audio dataset. Currently working to read and transform it. Looks like this repo has some pipeline to download at least the videos: `https://github.com/krantiparida/AudioSetZSL`.

To run download, execute `python google_dataset_proc/main.py`. For now will download only booms

### Youtube-dl Troubleshooting
https://stackoverflow.com/questions/63816790/youtube-dl-error-youtube-said-unable-to-extract-video-data