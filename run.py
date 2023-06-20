import os
import json
import time
import torch
import whisper
import warnings
import numpy as np
import pandas as pd
import soundcard as sc
import streamlit as st
from datetime import datetime
from collections import Counter
from janome.tokenizer import Tokenizer
warnings.simplefilter('ignore')

json_file_path = 'config.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

SAMPLE_RATE = data["SAMPLE_RATE"]
INTERVAL = data["INTERVAL"]
BUFFER_SIZE = data["BUFFER_SIZE"]

word_counter_noun, word_counter_verb, word_counter_adj, word_counter_interjection = st.columns(4)
with word_counter_noun:
    wc_noun = st.empty()
with word_counter_verb:
    wc_verb = st.empty()
with word_counter_adj:
    wc_adj = st.empty()
with word_counter_interjection:
    wc_interjection = st.empty()

st.subheader('検出テキスト')
result_text = st.empty()
col_left, col_right = st.columns(2)
with col_left:
    st.subheader('文字お越しデータ')
with col_right:
    btn_download_dataframe = st.empty()
result_dataframe = st.empty()

data = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = data

print('Loading model...')
placeholder = st.empty()
with placeholder.container():
    with st.spinner('Wait for it...'):
        model = whisper.load_model("large-v2", device="cpu")
        _ = model.half()
        _ = model.cuda()
        for m in model.modules():
            if isinstance(m, whisper.model.LayerNorm):
                m.float()
    st.success('Done!')
    time.sleep(2)
placeholder.empty()
print('Done')

# Define a function to update the displayed text
def update_result_text(text):
    result_text.code(text)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(st.session_state.df)
btn_download_dataframe.download_button(
    label="Press to Download",
    data=csv,
    file_name="file.csv",
    mime='text/csv',
)

# Janomeのインスタンスを作成
tokenizer = Tokenizer()
# 分かち書きと品詞別の出現頻度を格納する辞書
name_keys = ['名詞', '助詞','記号', '接続詞', '動詞', '形容詞', '助動詞', '副詞', '連体詞', '感動詞', '接頭詞', '接尾辞']
word_freq = {key: Counter() for key in name_keys}

# データフレームの各行に対して分かち書きと出現頻度の処理を行う関数
def count_word_freq(row):
    # 分かち書き
    tokens = tokenizer.tokenize(row) 
    # 品詞別に出現頻度をカウント
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]  # 品詞の取得
        if pos in word_freq:
            word_freq[pos][token.base_form] += 1
        else:
            word_freq[pos] = Counter({token.base_form: 1})

i, max_value = 0, 50
data_editor_dict = {}
with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
    options = whisper.DecodingOptions()
    n = 0
    b = np.ones(100) / 100
    while True:
        while n < SAMPLE_RATE * INTERVAL:
            data = mic.record(BUFFER_SIZE)
            audio[n:n + len(data)] = data.reshape(-1)
            n += len(data)

        # Find silent periods
        m = n * 4 // 5
        vol = np.convolve(audio[m:n] ** 2, b, 'same')
        m += vol.argmin()
        input_audio = audio[:m]

        if (input_audio ** 2).max() > 0.001:
            i += 1
            input_audio = whisper.pad_or_trim(input_audio)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(input_audio).to(model.device)

            # Detect the spoken language
            _, probs = model.detect_language(mel)

            # Decode the audio
            result = whisper.decode(model, mel, options)

            if max(probs, key=probs.get) == 'ja':
                # Call the update_result_text function to update the displayed text
                update_result_text(result.text)

                st.session_state.df = st.session_state.df.append({
                    'Datetime': datetime.now().time().strftime('%H:%M:%S'),
                    'Text': result.text,
                }, ignore_index=True)

                result_dataframe.dataframe(st.session_state.df, use_container_width=True)

                # Update the word frequency counts and word cloud for each row
                count_word_freq(result.text)

                data_editor_dict['名詞'] = dict(sorted(word_freq['名詞'].items(), key=lambda x: x[1], reverse=True))
                wc_noun.data_editor(
                    {
                        "name_key": list(data_editor_dict['名詞'].keys()), 
                        "name_count": list(data_editor_dict['名詞'].values())
                    },
                    column_config={
                        "name_key": "名詞",
                        "name_count": st.column_config.ProgressColumn(
                            "回数",
                            format="%f", max_value=max_value, width="small",
                        ),
                    },
                    key='noun{}'.format(i), hide_index=True,
                )

                data_editor_dict['動詞'] = dict(sorted(word_freq['動詞'].items(), key=lambda x: x[1], reverse=True))
                wc_verb.data_editor(
                    {
                        "name_key": list(data_editor_dict['動詞'].keys()), 
                        "name_count": list(data_editor_dict['動詞'].values())
                    },
                    column_config={
                        "name_key": "動詞",
                        "name_count": st.column_config.ProgressColumn(
                            "回数",
                            format="%f", max_value=max_value, width="small",
                        ),
                    },
                    key='verb{}'.format(i), hide_index=True,
                ) 

                data_editor_dict['形容詞'] = dict(sorted(word_freq['形容詞'].items(), key=lambda x: x[1], reverse=True))
                wc_adj.data_editor(
                    {
                        "name_key": list(data_editor_dict['形容詞'].keys()), 
                        "name_count": list(data_editor_dict['形容詞'].values())
                    },
                    column_config={
                        "name_key": "形容詞",
                        "name_count": st.column_config.ProgressColumn(
                            "回数",
                            format="%f", max_value=max_value, width="small",
                        ),
                    },
                    key='adj{}'.format(i), hide_index=True,
                )

                data_editor_dict['感動詞'] = dict(sorted(word_freq['感動詞'].items(), key=lambda x: x[1], reverse=True))                          
                wc_interjection.data_editor(
                    {
                        "name_key": list(data_editor_dict['感動詞'].keys()), 
                        "name_count": list(data_editor_dict['感動詞'].values())
                    },
                    column_config={
                        "name_key": "感動詞",
                        "name_count": st.column_config.ProgressColumn(
                            "回数",
                            format="%f", max_value=max_value, width="small",
                        ),
                    },
                    key='interjection{}'.format(i), hide_index=True,
                )

        audio_prev = audio
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        audio[:n-m] = audio_prev[m:n]
        n = n-m