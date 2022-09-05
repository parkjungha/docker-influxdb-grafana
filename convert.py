# coding: utf-8
import sys
import numpy as np
import pandas as pd

# np['time'] = (1656633600 +df['sequence'][t]*900)*1000000000

#급기_습도, 급기_온도, 외기_온도, 환기_온도, 환기_습도

df = pd.read_csv("./cdata.csv",sep=",",encoding="cp949")

lines = ["prediction_model"
         + ","
         + "data=" + "\"HVU1_급기_습도\""
         + " "
         + "real_data="+str(df["R_HVU1_급기_습도"][d])
         + " "
         + str((1656633600 +df['sequence'][d]*900)*1000000000) #start timestamp = 1656633600
         for d in range(2050)
         ]
thefile = open('./급기_습도_R.txt','w')

for item in lines:
    thefile.write("%s\n"%item)

lines2 = ["prediction_model"
         + ","
         + "data=" + "\"HVU1_급기_습도\""
         + " "
         + "prediction_result="+str(df["HVU1_급기_습도"][d])
         + " "
         + str((1656633600 +df['sequence'][d]*900)*1000000000)
         for d in range(len(df))
         ]

thefile2 = open('./급기_습도_P.txt','w')

for item in lines2:
    thefile2.write("%s\n"%item)