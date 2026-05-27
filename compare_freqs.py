import pandas as pd


df_admul = pd.read_csv(f"AdMul_pred/fp_counts_vuachristof_EN_DE.csv")
df_melbert = pd.read_csv(f"MelBERT_pred/fp_counts_vuachristof_EN_DE.csv")

def calc_diff(df):
    df["diff"] = df["count"]-df["freq2"]
    return df
 
def lookup_freq(df):
    df["freq2"] = df_admul[df_admul["word"]==df["word"]]["count"].values
    if len(df["freq2"]) > 0:
        df["freq2"] = df["freq2"][0]
    else:
        df["freq2"] = 0
    return df

def lookup_freq_reverse(df):
    df["freq2"] = df_melbert[df_melbert["word"]==df["word"]]["count"].values
    if len(df["freq2"]) > 0:
        df["freq2"] = df["freq2"][0]
    else:
        df["freq2"] = 0
    return df

def get_missing(df1, df2):
    count = len(df2) + 1
    for i, row in df1.iterrows():
        if row["word"] not in df2["word"].values:
            #print(row["word"])
            df3 = pd.DataFrame({"word":row["word"], "count":0, "freq2":row["count"]}, index=[count])
            df2 = pd.concat([df2, df3], axis=0)
            count += 1
    return df2

df_melbert = df_melbert.apply(lookup_freq, axis=1)
#print(len(df_christof))
#df_christof = get_missing(df_vuachristof, df_christof)
#print(len(df_christof))
df_melbert = df_melbert.apply(calc_diff, axis=1)
df_melbert.to_csv("fp_comparison_melbert_admul.csv")
df_admul = df_admul.apply(lookup_freq_reverse, axis=1)
df_admul = df_admul.apply(calc_diff, axis=1)
df_admul.to_csv("fp_comparison_admul_melbert.csv")