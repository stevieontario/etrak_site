from PIL import Image
import streamlit as st
import requests
import numpy as np
import pandas as pd
#import streamlit_lottie as stl
import os

cwd = os.getcwd()
print('cwd: ', cwd)
path_parent = os.path.dirname(cwd)
relative_path = '/data/'
full_path = path_parent+relative_path
print('full path: ',full_path)


# https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title='emissionTrak home', page_icon='📈', layout='wide')

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


heater_coding = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_FCYi2l.json')
temp_coding = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_m4znnezt.json')
hotPerson_coding = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_medwe7cs.json')
st.write('Cryo Systems Equipment primer on cryo freezing: http://www.cryobrain.com/nitrogen-vs-carbon-dioxide/. Also see Americold temp-controlled REIT, competitor of Lineage Logistics')
#stl.st_lottie(hotPerson_coding, width=200)
# --- WHAT I DO ---
with st.container():
    st.write('---')
    left_column, right_column = st.columns(2)
    with left_column:
        st.header('What I do')
        what_i_do = 'I advise clients on navigating the strategic features of their operating environment. I draw from decades of experience evaluating and observing technology in action, plus the mountains of data that are available for every dimension of the problem set, in the service of client goals. When goals and data clash, as they sometimes do, I help the client develop productive ways forward. When the clash is less important than the politics, as it often is, the challenge is a bit more subtle. I help the client steer through these treacherous waters.'
        st.write(what_i_do)
    with right_column:
        st.header('How I do it')
        df = pd.read_csv(full_path+'ECOMPCTNSA.csv')
        df = df.set_index('DATE')
        df.columns = ['e-commerce % change\n(not seasonally adjusted)']
        st.dataframe(df)


