import os
import re
from bert import Ner
import json

model_file = "ner_model_v7_10e"

model = Ner(model_file)

in_dir = "../emailbodies_with_EP_words_cleaned_"

##

out_file = "autolabeled_v7.jsonl"

jsonl = open(out_file,"w")

id = 0

x = {"id":id,"data":"","label":[],"tep":"$0"}

for file in os.listdir(in_dir):
    print(file)
    data = ""
    word_index = ""
    label = []
    id += 1
    tep = []
    in_file = os.path.join(in_dir,file)
    try:
        with open(in_file,"r") as eb:
            text = eb.read()
    except:
        continue
    index = text.lower().find("expiring premium")
    content = text[index-100:min(index+700,len(text))]
    pred = model.predict(content)
    for i in range(len(pred)):
        word = pred[i]["word"]
        l = index-100+len(data)
        data += word + " "
        h = l+len(word)

        tag = pred[i]["tag"]

        if tag != "O":
            label.append([l,h,tag])

        # if tag == "T":
        #     print(word)
        #     tep = word

    data = text[:index-100] + data + text[min(index-100+len(data),len(text)):]
    x["id"] = id
    x["data"] = data
    x["label"] = label
    # if len(tep) == 1:
    #     x["tep"] = tep[0]

    json.dump(x,jsonl)
    jsonl.write("\n")

jsonl.close()
