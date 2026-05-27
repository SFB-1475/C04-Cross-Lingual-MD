import lxml.etree as ET
import pandas as pd
from pathlib import Path

def print_tsv(sent_dict, sense_dict, inv_dict, lang, mode):
    outfile = open(f"{mode}_3_{lang}.tsv", "w", encoding="utf-8")
    outfile.write("sent_id\ttarget_id\ttarget_loc\tlemma\tpos\tsentence\tlabel\n")
    for sent_id in sent_dict.keys():
        sent_text = sent_dict[sent_id][0]
        for word in sent_dict[sent_id][1]:
            word_text = word[0]
            word_id = word[1]
            lemma = word[2]
            pos = word[3]
            word_index = word[4]
            possible_senses = inv_dict[(lemma.lower().replace(" ", "_"), pos)]
            if len(possible_senses) > 3:
                gold_sense = sense_dict[word_id]
                if possible_senses[0] == gold_sense:
                    label = "1"
                else:
                    label = "0"
                outfile.write(f"{sent_id}\t{word_id}\t{word_index}\t{lemma}\t{pos}\t{sent_text}\t{label}\n")
    outfile.close()



def get_inventory(invfile):
    inventory_dict = {}
    lines = invfile.readlines()
    for line in lines:
        line = line.strip().split("\t")
        key = line[0].split("#")
        inventory_dict[(key[0], key[1])] = line[1:] 
    return inventory_dict

def get_gold(goldfile):
    sense_dict = {}
    lines = goldfile.readlines()
    for line in lines:
        line = line.strip().split()
        sense_dict[line[0]] = line[1]
    return sense_dict

def read_xml(file):
    sent_dict = {}
    tree = ET.parse(file)
    root = tree.getroot()
    sents = root.findall(".//sentence")
    for sent in sents:
        sent_toks = [tok.strip() for tok in sent.itertext() if tok.strip()]
        sent_text = " ".join(sent_toks)
        sent_id = sent.attrib["id"]
        words = sent.findall(".//instance")
        sent_dict[sent_id] = [sent_text, []]
        for word in words:
            word_id = word.attrib["id"]
            lemma = word.attrib["lemma"]
            pos = word.attrib["pos"]
            word_text = word.text
            #
            try: 
                word_index = sent_text.split().index(word_text)
            except ValueError:
                for i, tok in enumerate(sent_toks):
                    if word == tok:
                        word_index == i
            sent_dict[sent_id][1].append((word_text, word_id, lemma, pos, word_index))
    return sent_dict


def main():
    for lang in ["en", "de"]:
        inventory = open(f"xl-wsd/inventories/inventory.{lang}.txt", "r", encoding="utf-8")
        inventory_dict = get_inventory(inventory)
        for mode in ["training", "evaluation"]:
            if mode == "training":
                sentfile = open(f"xl-wsd/{mode}_datasets/semcor_{lang}/semcor_{lang}.data.xml", "r", encoding="utf-8")
                gold_labels = open(f"xl-wsd/{mode}_datasets/semcor_{lang}/semcor_{lang}.gold.key.txt", "r", encoding="utf-8")
            else:
                sentfile = open(f"xl-wsd/{mode}_datasets/dev-{lang}/dev-{lang}.data.xml", "r", encoding="utf-8")
                gold_labels = open(f"xl-wsd/{mode}_datasets/dev-{lang}/dev-{lang}.gold.key.txt", "r", encoding="utf-8")
            gold_sense_dict = get_gold(gold_labels)
            print(inventory_dict)
            sent_dict = read_xml(sentfile)
            print_tsv(sent_dict, gold_sense_dict, inventory_dict, lang, mode)

main()