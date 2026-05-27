import argparse
import pandas as pd
from sklearn.metrics import classification_report

def correct_label(df):
    #print(df)
    if df["similarity"].lower().startswith("ja"):
        df["final_label"] = 1
    return df

def read_labels_llm(test_data, predsfile):
    mrw_labels = []
    mrw_token = []
    mrw_full_sent = []
    indices = []
    df_preds = pd.read_csv(predsfile)
    df_preds = df_preds.apply(correct_label, axis=1)
    print(df_preds)
    for line in test_data.readlines()[1:]:
        line = line.split("\t")
        token_index = str(int(line[5])+1)
        indices.append(line[0]+ " " +token_index)
        mrw_label = line[1]
        mrw_labels.append(mrw_label)
        mrw_full_sent.append(line[2])
        #token_index.append(line[-1])
        #print(int(line[-2]))
        mrw_token.append(line[2].split()[int(line[5])])
    print(len(mrw_labels))

    preds = df_preds["final_label"].values
    preds = [str(pred) for pred in preds]
    print(preds)
    print(len(preds))
    print(classification_report(mrw_labels, preds,digits=4))
    return indices, mrw_labels, preds, mrw_full_sent, mrw_token


def read_labels(test_data, predsfile):
    mrw_labels = []
    preds = []
    mrw_token = []
    mrw_full_sent = []
    indices = []
    #token_index = []
    for line in test_data.readlines()[1:]:
        line = line.split("\t")
        token_index = str(int(line[5])+1)
        indices.append(line[0]+ " " +token_index)
        mrw_label = line[1]
        mrw_labels.append(mrw_label)
        mrw_full_sent.append(line[2])
        #token_index.append(line[-1])
        #print(int(line[-2]))
        mrw_token.append(line[2].split()[int(line[5])])
    for line in predsfile.readlines():
        preds.append(line.strip()[-1])
    print(len(preds))
    print(len(mrw_labels))
    print(classification_report(mrw_labels, preds, digits=4))
    return indices, mrw_labels, preds, mrw_full_sent, mrw_token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang")
    parser.add_argument("--target_lang")
    parser.add_argument("--data")
    parser.add_argument("--model")
    args = parser.parse_args()
    if not args.model == "LLM":
        predfile = open(f"{args.model}_pred/{args.data}_{args.source_lang}_{args.target_lang}_preds_raw.txt", "r", encoding="utf-8")
    else:
        predfile = f"LLM_pred/data_annotated_mipvu_christof_Llama-3.1-8B-Instruct_{args.target_lang}.csv"
    if args.target_lang == "DE":
        goldfile = open(f"test_christof_DE.tsv", "r", encoding="utf-8")
    elif args.target_lang == "EN":
        goldfile = open(f"test_christof_EN.tsv", "r", encoding="utf-8")
    if args.model == "LLM":
        indices, mrw_labels, preds, mrw_full_sent, mrw_token = read_labels_llm(goldfile, predfile)
    else:
        indices, mrw_labels, preds, mrw_full_sent, mrw_token = read_labels(goldfile, predfile)
    df = pd.DataFrame({"index":indices, "word":mrw_token, "sentence":mrw_full_sent, "pred":preds, "gold":mrw_labels})
    df.to_csv(f"{args.model}_pred/comparison_{args.data}_{args.source_lang}_{args.target_lang}.tsv")
    df_fn = df[(df["pred"]=="0")&(df["gold"]=="1")]
    df_fp = df[(df["pred"]=="1")&(df["gold"]=="0")]
    df_fn_counts = df_fn["word"].value_counts()
    df_fp_counts = df_fp["word"].value_counts()
    df_fn.to_csv(f"{args.model}_pred/fn_{args.data}_{args.source_lang}_{args.target_lang}.csv")
    df_fp.to_csv(f"{args.model}_pred/fp_{args.data}_{args.source_lang}_{args.target_lang}.csv")
    df_fn_counts.to_csv(f"{args.model}_pred/fn_counts_{args.data}_{args.source_lang}_{args.target_lang}.csv")
    df_fp_counts.to_csv(f"{args.model}_pred/fp_counts_{args.data}_{args.source_lang}_{args.target_lang}.csv")
main()