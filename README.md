# ROI decoder

<img width="1065" alt="Screen Shot 2022-09-15 at 2 12 52 AM" src="https://user-images.githubusercontent.com/7505975/190500757-f10a3bf4-984c-46b9-906c-fa0fc89cc423.png">

Streamlit app that computes a coarse estimate of stimuli tuning for calcium imaging data -- can be used in a semi-online fashion, during recording sessions to find responsive areas to to record further from. 

Takes a `tif` stack and a `csv` file with stimuli information and computes localized logistic regression based decoders to see which regions are sensitive to which stimulus.

## Seutp

Install prequisites:
```
pip install -r requirements.txt
```

## How to run

```
streamlit run --server.fileWatcherType none streamlit_app.py
```

You can specify a default recording to analyze with the arguments `--tifname` and `--tonefile`:
```
streamlit run --server.fileWatcherType none streamlit_app.py --tifname ./demodata/TSeries-07062022-001_rig__d1_512_d2_512_d3_1_order_F_frames_4000_.tif --tonefile ./demodata/roiscan1.csv
```

Then point your browser to `http://localhost:8501`

## How to use

Point the app to the tif stack you want to analyze, and the stimuli csv file. This file is just a list of `time,stim` pairs, e.g:
```
1.02E-03,5500
1.001021385,10000
2.001018763,22000
...
```
Where in this case `5500`, `10000` are tones presented to the animal. These are assumed to be discrete categories -- logistic regression is used to predict these categories. Time stamps are in seconds. Specify the FPS of the recording in the field to match these to the tif file.

Set decoder parameters on the right and hit Run!
 
