# Developed for predicting the tennis match outcomes

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle

# 0.66 %80

df = pd.read_csv('atp_matches_2020.csv',encoding="utf8")
df["date"] = pd.to_datetime(df["tourney_date"],format="%Y%m%d")

df.replace(('R', 'L'), (1, 0), inplace=True)

#print(df.info())

date = pd.to_datetime("22082020",format="%d%m%Y")

surface="Hard"              # Hard or Clay or Glass

ev_sahibi = 111575

deplasman = 105526

op_id = ev_sahibi           # e.g 105018
fav_id = deplasman


#op_rank_point= 1395

#fav_rank_point=1771

df_op = df[ (df["winner_id"] == op_id) | (df["loser_id"] == op_id)]


if(len(df_op.index)>0):
    df_op = df_op.tail(n=1)
    if(df_op["winner_id"].iloc[-1] == op_id):

        op_rank_point = df_op["winner_rank_points"].iloc[-1]
        op_hand = df_op["winner_hand"].iloc[-1]

    elif(df_op["loser_id"].iloc[-1] == op_id):

        op_rank_point = df_op["loser_rank_points"].iloc[-1]
        op_hand = df_op["loser_hand"].iloc[-1]

    else:
        print("PROBLEM")


if(op_hand != 0 and op_hand != 1):
    op_hand == 1  # 1 for Right, 0 for Left


df_fav = df[(df["winner_id"] == fav_id) | (df["loser_id"] == fav_id)]

if (len(df_fav.index) > 0):
    df_fav = df_fav.tail(n=1)
    if (df_fav["winner_id"].iloc[-1] == fav_id):

        fav_rank_point = df_fav["winner_rank_points"].iloc[-1]
        fav_hand = df_fav["winner_hand"].iloc[-1]

    elif (df_fav["loser_id"].iloc[-1] == fav_id):

        fav_rank_point = df_fav["loser_rank_points"].iloc[-1]
        fav_hand = df_fav["loser_hand"].iloc[-1]

    else:
        print("PROBLEM")

if (fav_hand != 0 and fav_hand != 1):
    fav_hand == 1  # 1 for Right, 0 for Left

print(op_rank_point,"!!!")

print(fav_rank_point)

datestart = date - pd.DateOffset(months=8)

sub = df[(df['date'] > datestart) & (df['date'] < date) & (df["winner_id"] == fav_id) & (df["surface"] == surface)]
fav_win = len(sub.index)

sub = df[(df['date'] > datestart) & (df['date'] < date) & (df["loser_id"] == fav_id) & (df["surface"] == surface)]
fav_lose = len(sub.index)

try:
        fav_ratio = fav_win / (fav_win + fav_lose)
except:
        fav_ratio=0.5


sub = df[(df['date'] > datestart) & (df['date'] < date) & (df["winner_id"] == op_id) & (df["surface"] == surface)]
op_win = len(sub.index)

sub = df[(df['date'] > datestart) & (df['date'] < date) & (df["loser_id"] == op_id) & (df["surface"] == surface)]
op_lose = len(sub.index)

try:
        op_ratio = op_win / (op_win + op_lose)
except:
        op_ratio=0.5

print(op_ratio)
print(fav_ratio)

model = tf.keras.models.load_model(
    ".\iamfortennis8.h5",
    custom_objects = {"Keras Layer" : hub.KerasLayer}
    )


x = model.predict(np.array([[op_rank_point,op_hand,op_ratio,fav_rank_point,fav_hand,fav_ratio]]))
print("\n",x[0][0])

model = pickle.load(open("LRtennis8.sav", "rb"))
x = model.predict(np.array([[op_rank_point,op_hand,op_ratio,fav_rank_point,fav_hand,fav_ratio]]))
print(x[0])


