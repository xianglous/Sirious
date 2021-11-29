import pandas as pd

df1 = pd.read_csv('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious/summarization_model/data/ted_dataset_kaggle/ted_main.csv',encoding='latin-1')
df2 = pd.read_csv('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious/summarization_model/data/ted_dataset_kaggle/transcripts.csv',encoding='latin-1')
df1 = df1[["description","url"]]
df2 = df2[["transcript", "url"]]


print(type(df2["url"]))


for url in df1["url"]:
    if url in df2["url"]:
        final_df = pd.DataFrame({'description':df1["description"],'transcipt':df2["transcipt"]})
        final_df.to_csv('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious/summarization_model/data/ted_dataset_kaggle/final.csv')