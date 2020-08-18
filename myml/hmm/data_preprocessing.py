import json

with open("./corpus/dh_msra.txt", 'r') as f:
    lines = f.readlines()
    tmp_texts, tmp_labels, all_texts, all_labels = [], [], [], []
    for line in lines:
        if line == '\n':
            all_labels.append(tmp_labels)
            all_texts.append(tmp_texts)
            tmp_texts, tmp_labels = [], []
        else:
            char, tag = line.split()
            tmp_labels.append(tag)
            tmp_texts.append(char)

labels, texts = [], []
with open("./corpus/train.txt", 'w') as f:
    for text, label in zip(all_texts, all_labels):
        labels.extend(label)
        texts.extend(text)
        f.write(str({"text": text, "label": label}))
        f.write("\n")

# 构造字符词典和标签词典
labels, texts = set(labels), set(texts)
char2idx = {text: idx + 1 for idx, text in enumerate(texts)}
label2idx = {label: idx for idx, label in enumerate(labels)}
char2idx["UNK"] = 0

with open("./corpus/char2idx.json", 'w') as f:
    f.write(json.dumps(char2idx, ensure_ascii=False))
with open("./corpus/label2idx.json", 'w') as f:
    f.write(json.dumps(label2idx, ensure_ascii=False))



