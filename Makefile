demo:
		streamlit run --server.fileWatcherType none streamlit_app.py --tifname ./demodata/TSeries-07062022-001_rig__d1_512_d2_512_d3_1_order_F_frames_4000_.tif --tonefile ./demodata/roiscan1.csv 

run:
		streamlit run --server.fileWatcherType none streamlit_app.py

#The option --server.fileWatcherType none is used to avoid the error 'OSError: [Errno 28] inotify watch limit reached'