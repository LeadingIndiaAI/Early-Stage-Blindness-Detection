import shutil as sh
import pandas as pd
df_train=pd.read_csv("C:\Users\Rohan\Downloads\aptos2019-blindness-detection\train.csv")
for index, row in df_train.iterrows():
    if row['diagnosis']==0:
        a=str(row['id_code'])+'.png'
        sh.move(a,'0')
    elif row['diagnosis']==1:
        a=str(row['id_code'])+'.png'
        sh.move(a,'1')
    elif row['diagnosis']==2:
        a=str(row['id_code'])+'.png'
        sh.move(a,'2')
    elif row['diagnosis']==3:
        a=str(row['id_code'])+'.png'
        sh.move(a,'3')
    elif row['diagnosis']==4:
        a=str(row['id_code'])+'.png'
        sh.move(a,'4')