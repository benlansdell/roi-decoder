import streamlit as st
from pathlib import Path

import os
import streamlit as st

def update_dir(key):
    #Get our choice 
    choice = st.session_state[key]
    print(choice)
    if os.path.isdir(os.path.join(st.session_state[key+'curr_dir'], choice)):
        st.session_state[key+'curr_dir'] = os.path.normpath(os.path.join(st.session_state[key+'curr_dir'], choice))
        files = sorted(os.listdir(st.session_state[key+'curr_dir']))
        files.insert(0, '..')
        files.insert(0, '.')
        st.session_state[key+'files'] = files

def st_file_selector(st_placeholder, path='.', label='Select a file/folder', key = 'selected'):
    if key+'curr_dir' not in st.session_state:
        base_path = '.' if path is None or path is '' else path
        base_path = base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
        base_path = '.' if base_path is None or base_path is '' else base_path
        st.session_state[key+'curr_dir'] = base_path

    else:
        base_path = st.session_state[key+'curr_dir']

    #Sanity check...
    if not os.path.exists(base_path):
        base_path = '.'

    files = sorted(os.listdir(base_path))
    files.insert(0, '..')
    files.insert(0, '.')
    st.session_state[key+'files'] = files
    print(st.session_state)   
    selected_file = st_placeholder.selectbox(label=label, 
                                        options=st.session_state[key+'files'], 
                                        key=key, 
                                        on_change = lambda: update_dir(key))
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    st_placeholder.write(os.path.abspath(selected_path))

    return selected_path
