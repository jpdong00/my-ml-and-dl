import torch
from torch import nn, optim

from transformer import Transformer

# 1. 数据准备
de = ['ich mochte ein bier',
      'Seit der Veröffentlichung ihrer Artikel ist der Goldpreis noch weiter gestiegen',
      'Jüngst erreichte er sogar ein Rekordhoch von 1300 Dollar']

en = ['i want a beer',
      'Since their articles appeared the price of gold has moved up still further',
      'Gold prices even hit a record-high $1300 recently']

# 数据样式：'ich mochte ein bier P', 'S i want a beer', 'i want a beer E'
# P: 填充符号
pad_token = ['P']
# S: 开始符号
start_token = ['S']
# E: 结束符号
end_token = ['E']

src_vocab = pad_token + list(set([word for sen in de for word in sen.split(' ')]))
src_vocab_size = len(src_vocab)
src_word2index = {word: idx for idx, word in enumerate(src_vocab)}
src_index2word = {idx: word for idx, word in enumerate(src_vocab)}
print('src_vocab_size: ', src_vocab_size)

tgt_vocab = pad_token + list(set([word for sen in en for word in sen.split(' ')])) + start_token + end_token
tgt_vocab_size = len(tgt_vocab)
tgt_word2index = {word: idx for idx, word in enumerate(tgt_vocab)}
tgt_index2word = {idx: word for idx, word in enumerate(tgt_vocab)}
print('tgt_vocab_size: ', tgt_vocab_size)

max_seq_len = 15
input_batch = []
for sen in de:
    words = sen.split(' ')
    pad_len = max_seq_len - len(words)
    if pad_len > 0:
        words += pad_token * pad_len
    input_batch.append([src_word2index[word] for word in words])

output_batch, target_batch = [], []
for sen in en:
    words = sen.split(' ')
    outputs, targets = start_token + words, words + end_token
    pad_len = max_seq_len - len(outputs)
    if pad_len > 0:
        outputs += pad_token * pad_len
        targets += pad_token * pad_len
        output_batch.append([tgt_word2index[word] for word in outputs])
        target_batch.append([tgt_word2index[word] for word in targets])

enc_inputs = torch.tensor(input_batch, dtype=torch.long)
dec_inputs = torch.tensor(output_batch, dtype=torch.long)
target_batch = torch.tensor(target_batch, dtype=torch.long)

# 2. 训练
model = Transformer(src_vocab_size, max_seq_len, tgt_vocab_size, max_seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(20):
    optimizer.zero_grad()
    # outputs: batch_size, max_sequence_length, tgt_vocab_size
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs.view(-1, tgt_vocab_size), target_batch.view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# 3. 测试
predict, _, _, _ = model(enc_inputs, dec_inputs)
predict = predict.data.max(-1)[1]
for i, sen in enumerate(predict):
    pred_sen = [tgt_index2word[word] for word in sen.tolist()]
    end_index = pred_sen.index("E")
    print(de[i] + ' --> ' + ' '.join(pred_sen[:end_index]))
