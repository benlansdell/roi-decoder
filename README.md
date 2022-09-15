# ROI decoder

<img width="1065" alt="Screen Shot 2022-09-15 at 2 12 52 AM" src="https://user-images.githubusercontent.com/7505975/190500757-f10a3bf4-984c-46b9-906c-fa0fc89cc423.png">

Streamlit app that computes a coarse estimate of stimuli tuning for calcium imaginge data -- can be used in a semi-online fashion, during recording sessions to find responsive areas to to record further from. 

Takes a `tif` stack and a csv with stimuli information and computes localized logistic regression based decoders to see which regions are sensitive to which stimulus.

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
