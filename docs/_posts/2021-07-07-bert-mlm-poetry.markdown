---
layout: post
title: "用BERT做古诗词MLM任务"
date: 2021-07-07 16:28:00 +0800
categories: NPL
---

<style type="text/css">
    div.output_area pre {
    margin: 0;
    padding: 1px 0 1px 0;
    border: 0;
    vertical-align: baseline;
    color: black;
    background-color: transparent;
    border-radius: 0;
    white-space: pre-wrap;
    }
    /* This class is for the output subarea inside the output_area and after
    the prompt div. */
    div.output_subarea {
    overflow-x: auto;
    padding: 0.4em;
    /* Old browsers */
    -webkit-box-flex: 1;
    -moz-box-flex: 1;
    box-flex: 1;
    /* Modern browsers */
    flex: 1;
    max-width: calc(100% - 14ex);
    }
    div.output_scroll div.output_subarea {
    overflow-x: visible;
}
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
pre {
    white-space: pre-wrap;
}

</style>

<a href="https://arxiv.org/abs/1810.04805">BERT</a>的全称是Bidirectional Encoder Representations from Transformers, 是目前最强大的自然语言模型。下面用几个简单的例子，来看看如何用BERT。

BERT用Masked Language Model(MLM)和Next Sequence Prediction（NSP)两个任务训练模型。MLM任务让程序通过上下文预测句子中被掩盖（Mask）的词语。NSP任务中AI模型判断两个句子是否是上下文关系。训练之后的模型，通过简单的fine tune可以用于其他任务，比如文本分类，情感分析等。

本文通过古诗词来学习如何对BERT做MLM任务训练。我们先安装需要用到的python模块

```python
pip install tensorflow transformers numpy zhconv scikit_learn
```

以下为需要import的模块以及一些全局常量：

```python
import numpy as np
import json
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils

batch_size = 32 #训练的batch_size
input_dim = 32  # 输入句子的长度
epochs = 3      # 训练的epochs
val_split = 0.1 # 训练和验证的数据比

max_tang_index = 57     # 唐诗文件的数目
max_song_index = 254    # 宋诗文件的数目

```

# 预处理

训练用的数据采用Github上开源的<a href="https://github.com/chinese-poetry/chinese-poetry">中华古诗词库</a>。我们用jason目录下的唐诗和宋诗，将所有格式为poet.tang.xxx000.json和peot.song.xxx000.json的文件下载到本地。加载json文件后需要用zhconv模块将繁体字转为简体字。

```python
from zhconv import convert

data_root = './datasets/' # 存储诗词的目录

def load_json(index, type):
  path = os.path.join(data_root, str.format('poet.{0}.{1}.json', type, index * 1000))
  with open(path, 'r', encoding='utf8') as f:
    return json.load(f)

def convert_to_simplify(json_obj):
  json_obj['author'] = convert(json_obj['author'], 'zh-cn')
  json_obj['title'] = convert(json_obj['title'], 'zh-cn')
  json_obj['paragraphs'] = [convert(p, 'zh-cn') for p in json_obj['paragraphs']]
  return json_obj

peot_list = load_json(2, 'tang')
print(len(peot_list))
print(convert_to_simplify(peot_list[0]))

```
以上输出：
<div class="output_subarea output_text">
<pre>
1002
[{'author': '蓋嘉運', 'paragraphs': ['聞道黃花戍，頻年不解兵。', '可憐閨裏月，偏照漢家營。'], 'title': '雜曲歌辭 伊州 歌第三', 'id': '71700e1e-f443-4d15-8ad9-0b7d116afde5'}, {'author': '蓋嘉運', 'paragraphs': ['千里東歸客，無心憶舊遊。', '挂帆游白水，高枕到青州。'], 'title': '雜曲歌辭 伊州 歌第四', 'id': '3e9f7376-b439-4ae5-a7cc-c7b987b19317'}]
</pre>
</div>

以下是两个utility方法，filter_non_poetry_part将诗句中的注释以及非汉字部分去掉，random_poetry_sentences用来随机选取诗句。

```python

def is_ch(ch):
  if '\u4e00' <= ch <= '\u9fff':
    return True
  else:
    return False

def filter_non_poetry_part(text):
  filtered  = ''

  in_parenthsis = 0
  part = 0
  for ch in text.strip():
    if ch == '（':
      in_parenthsis += 1
    elif ch == '）':
      in_parenthsis -= 1
    if in_parenthsis:
      continue
    
    filtered += ch
    if ch == '，':
      part += 1

    elif ch in ['。', '？', '！']:
      if part != 1:
        filtered = ''
      break

    elif not is_ch(ch):
      filtered = ''
      break
    
  return filtered if part == 1 else ''

def random_poetry_sentences(size, type = 'tang'):
  max_index = max_tang_index if type == 'tang' else max_song_index
  indices = np.arange(1, max_index + 1)
  np.random.shuffle(indices)
  sentences = []

  for index in indices:
    peotry_list = load_json(index, type)
    for poetry in peotry_list:
      if not poetry['paragraphs']:
        continue

      convert_to_simplify(poetry)
    for text in poetry['paragraphs']:
        if random.random() >= size:
            continue
        text = filter_non_poetry_part(text)
        if len(text) >= 10 and len(text) <= input_dim - 2:
            sentences.append(text)

  return sentences

```

我们随机选取70%的诗句用来训练，然后用train_test_split方法将数据分为训练部分和验证部分。

```python
from sklearn.model_selection import train_test_split

def load_poetry_sentences(tang_poetry_size, song_poetry_size):
  tang_sentences = random_poetry_sentences(tang_poetry_size, 'tang')
  song_sentences = random_poetry_sentences(song_poetry_size, 'song')

  all_poetry_sentences = tang_sentences + song_sentences

  return train_test_split(all_poetry_sentences, test_size = val_split)

tr_poetry_sentences, val_poetry_sentences = load_poetry_sentences(tang_poetry_size, song_poetry_size)

print('no. of train data =', len(tr_poetry_sentences))
print(tr_poetry_sentences[:10])

print('no. of validation data =', len(val_poetry_sentences))
print(val_poetry_sentences[:10])
```

以上输出

```
no. of train data = 824702
['凤凰相对盘金缕，牡丹一夜经微雨。', '采尽舆情将入告，谋谟谁得见精深。', '飞萤忽点衣，小立闻荷香。', '从今朝谒得休暇，勿使文会多间阔。', '北客未识猫头笋，此来逢君欣得之。', '饮罢月斜吟兴动，挥毫风露一天寒。', '世间八月十五夜，何处楼台得月多。', '昔在后宫时，几见君王面。', '云汉何渺茫，远过衡山阳。', '木落山凋水见涯，感时短发半苍华。']
no. of validation data = 91634
['夔子山高峡天阔，骚人宅近刚肠悲。', '掩卷勿重陈，恸哭伤人气。', '痴儿呵冻为渠赋，费尽壶中墨客卿。', '幽讨平生如此少，清闲今日许谁知。', '相茵朝以登，胡骑夕可逐。', '平生事靖退，墨头自要津。', '潮满江津猿鸟啼，荆夫楚语飞蛮桨。', '维时南风薰，木叶晃繁碧。', '吴儿便操舟，衣服皆楚制。', '稽首大慈，如大地载。']
```

# 生成训练数据

对每一句诗句，随机选取一个字替换为'[MASK]'。如果被替的字是冷僻字，即在分词器的字典里找不到该字，则换一个随机字。分词器采用transformer模块中的BertTokenizer。'bert-base-chinese'是已训练过的基于中文的BERT模型。

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def to_masked_sentence(sentences):
  masked_sentences = []
  masked_words = []

  for sen in sentences:
    for i in range(10):
      w_index = random.randint(0, len(sen) - 1)
      if sen[w_index] not in ['。', '，', '？', '！'] and tokenizer.convert_tokens_to_ids(sen[w_index]) != tokenizer.unk_token_id:    
        break
    else:
      print('cannot find masked word for:', sen)

    masked_words.append(sen[w_index])
    masked_sentences.append(sen[:w_index] + '[MASK]' + sen[w_index + 1:])

  return masked_sentences, masked_words

```

我们定义一个用来提供训练数据的MaskedPoetryDataSet类。

```python

class MaskedPoetryDataSet(keras.utils.Sequence):
  def __init__(self, sentences, input_dim, batch_size):
    self.sentences = sentences
    self.input_dim = input_dim
    self.batch_size = batch_size
  
  def __len__(self):
    return (len(self.sentences) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    end_pos = min((index + 1) * self.batch_size, len(self.sentences))
    batch = self.sentences[index * self.batch_size: end_pos]
    masked_sentences, masked_words = to_masked_sentence(batch)
    input_ids = convert_texts_to_input_ids(masked_sentences)
    masked_words_ids = tokenizer.convert_tokens_to_ids(masked_words)

    return input_ids, np.array(masked_words_ids)
  
  def on_epoch_end(self):
    random.shuffle(self.sentences)


train_ds = MaskedPoetryDataSet(tr_poetry_sentences, input_dim = input_dim, batch_size = batch_size)
val_ds = MaskedPoetryDataSet(val_poetry_sentences, input_dim = input_dim, batch_size = batch_size)

```

# 建立模型

TFBertForMaskedLM是transformers提供的基于tensorflow的bert MLM模型。TFBertForMaskedLM的输出结果包含logits属性，形状为 (batch_size, input_dim, vacb_size)，其中vacb_size为词库的大小。经过一次softmax计算，logits转化为预测结果的分布概率。

```python

from transformers import TFBertForMaskedLM, BertConfig

config = BertConfig.from_pretrained('bert-base-chinese',
                                    output_attentions = False,
                                    output_hidden_states = False,
                                    use_cache = True,
                                    return_dict = True)

bert_mlm = TFBertForMaskedLM.from_pretrained('bert-base-chinese', config = config)

def build_model(bert_mlm):
  # 生成输入
  input_ids = keras.Input(shape = (input_dim, ), dtype='int32', name = 'input_ids')
  attention_mask = tf.where(input_ids == tokenizer.pad_token_id, 0.0, 1.0)
  # 调用bert模型
  output = bert_mlm([input_ids, attention_mask])
  # 选取[MASK]位置对应的logits
  mask_positions = tf.gather(tf.where(input_ids == tokenizer.mask_token_id), indices = 1, axis = -1)
  output = tf.gather(output.logits, mask_positions, axis = 1, batch_dims = 1)
  # 生成模型
  model = keras.models.Model(input_ids, output)
  
  return model

model = build_model(bert_mlm)

```
# 训练和结果

以下是用来预测以及显示预测结果的方法。如果概率最大的前topk个预测包含正确结果，则我们认为预测成功。

```python

def predict(bert_mlm, masked_sentence, topk = 1):
  # 将文本转成int数组
  input_ids = convert_texts_to_input_ids(masked_sentence)
  # 记录被遮盖的位置
  masked_pos = np.where(input_ids == tokenizer.mask_token_id)[1]
  # 生成注意力掩码
  attention_mask = (input_ids != tokenizer.pad_token_id).astype('int32')
  # 调用bert模型
  output = bert_mlm([input_ids, attention_mask])
  # 选取[MASK]所在的位置的结果
  logits = tf.gather(output.logits, masked_pos, axis = 1, batch_dims = 1)
  # 计算概率
  prob = tf.nn.softmax(logits, axis = -1)
  # 选取topk最大概率
  indices = tf.argsort(prob, axis = -1, direction = 'DESCENDING')
  indices = indices[:, :topk]
  flat_indices = tf.reshape(indices, shape = (indices.shape[0] * topk, )).numpy()
  
  # 预测结果转成文本
  tokens = np.array(tokenizer.convert_ids_to_tokens(flat_indices))
  tokens = np.reshape(tokens, indices.shape)
  return tokens.tolist(), tf.gather(prob, indices, axis = 1, batch_dims = 1).numpy().tolist()

def compute_accuracy(masked_words, predict_words):
    is_correct = [1 if masked_words[i] in pred else 0 for i, pred in enumerate(predict_words)]
    return sum(is_correct) / len(is_correct)

def show_prediction(masked_sentences, masked_words, predict_words, predict_prob):
  correct_predicts = []
  wrong_predicts = []

  for sen, expected, retured, prob in zip(masked_sentences, masked_words, predict_words, predict_prob):
    if expected in retured:
      correct_predicts.append((sen, expected, list(zip(retured, prob))))
    else:
      wrong_predicts.append((sen, expected, list(zip(retured, prob))))

  print('correct predicts:')
  for sen in correct_predicts:
    print((sen[0].replace('[MASK]', '__'), sen[1]), '\tprediction: ', sen[2])

  print()
  print('incorrect predicts:')
  for sen in wrong_predicts:
    print((sen[0].replace('[MASK]', '__'), sen[1]), '\tprediction: ', sen[2])

```

在训练之前，我们选取一小部分数据，输出模型的预测结果。该结果用来与训练后的结果做对比。我们选取topk=2，如果概率最高的前两个预测包含正确结果，即为预测成功。

```python
masked_sentences_sample, masked_words_sample = to_masked_sentence(val_poetry_sentences[:100])
predict_words_pretrained, predict_prob_pretrained = predict(bert_mlm, masked_sentences_sample, topk = 2)
print('accuracy =', compute_accuracy(masked_words_sample, predict_words_pretrained))
show_prediction(masked_sentences_sample, masked_words_sample, predict_words_pretrained, predict_prob_pretrained)
```
输出：

```python
accuracy = 0.26
correct predicts:
('掩卷勿重陈，恸哭伤__气。', '人') 	prediction:  [('人', 0.07060227543115616), ('元', 0.039788197726011276)]
('幽讨平__如此少，清闲今日许谁知。', '生') 	prediction:  [('生', 0.7154790759086609), ('时', 0.05146220326423645)]
('稽首__慈，如大地载。', '大') 	prediction:  [('慈', 0.13907204568386078), ('大', 0.12231048196554184)]
('苍苔复阁连荒草，乔__参云挂老藤。', '木') 	prediction:  [('木', 0.32727381587028503), ('树', 0.14084425568580627)]
('频岁驱驰行万__，还家幸矣敢言疲。', '里') 	prediction:  [('里', 0.9636253118515015), ('载', 0.002551188925281167)]
...

incorrect predicts:
('夔子山高峡天阔，骚人宅近__肠悲。', '刚') 	prediction:  [('断', 0.47469690442085266), ('肝', 0.0511016882956028)]
('痴儿呵冻为__赋，费尽壶中墨客卿。', '渠') 	prediction:  [('诗', 0.2743428647518158), ('君', 0.1260828971862793)]
('相茵朝以登，胡骑__可逐。', '夕') 	prediction:  [('不', 0.4636416435241699), ('未', 0.06769382208585739)]
('平生事靖退，墨__自要津。', '头') 	prediction:  [('客', 0.21530413627624512), ('迹', 0.07961680740118027)]
('潮满江津猿鸟啼，荆夫__语飞蛮桨。', '楚') 	prediction:  [('无', 0.21794617176055908), ('不', 0.08835069835186005)]

...

```

输入100个句子，得到26个正确的预测，正确率为0.26。

下面我们来做训练。Bert模型训练结果对学习速率非常敏感，一般只能用很小的速率学习。论文采用了先上升后递减的速率，速率峰值为3e-5。我们采用论文相同的学习速率配置。

```python
from transformers import create_optimizer

optimizer, lr_schedule = create_optimizer(
                          init_lr = 3e-5,
                          weight_decay_rate = 0.01,
                          num_train_steps = int(len(train_ds) * epochs),
                          num_warmup_steps = int(0.1 * len(train_ds)))

model.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(train_ds, epochs = epochs, validation_data = val_ds)

```
我们训练了3个epoch。在Google colab上用一个GPU训练一个epoch大概需要1个小时。训练后，我们再次看看预测结果

```python
predict_words_trained, predict_prob_trained = predict(bert_mlm, masked_sentences_sample, topk = 2)
print('accuracy =', compute_accuracy(masked_words_sample, predict_words_trained))
show_prediction(masked_sentences_sample, masked_words_sample, predict_words_trained, predict_prob_trained)
```

正确率有了比较明显的提升，从0.26提高到0.44

```python
accuracy = 0.44
correct predicts:
('幽讨平__如此少，清闲今日许谁知。', '生') 	prediction:  [('生', 0.9553178548812866), ('时', 0.03859643638134003)]
('相茵朝以登，胡骑__可逐。', '夕') 	prediction:  [('夜', 0.15661050379276276), ('夕', 0.15394945442676544)]
('吴儿便操舟，衣服__楚制。', '皆') 	prediction:  [('皆', 0.08485011756420135), ('为', 0.07700314372777939)]
('稽首__慈，如大地载。', '大') 	prediction:  [('大', 0.4974978566169739), ('慈', 0.08413518220186234)]
('苍苔复阁连荒草，乔__参云挂老藤。', '木') 	prediction:  [('木', 0.959535539150238), ('树', 0.017344415187835693)]
('__典久传祠太乙，竺坟亦说会夫人。', '汉') 	prediction:  [('汉', 0.12053366750478745), ('禹', 0.06589178740978241)]
...

incorrect predicts:
('夔子山高峡天阔，骚人宅近__肠悲。', '刚') 	prediction:  [('断', 0.6204995512962341), ('离', 0.054959289729595184)]
('掩卷勿重陈，恸哭伤__气。', '人') 	prediction:  [('士', 0.12313748151063919), ('志', 0.07649099081754684)]
('痴儿呵冻为__赋，费尽壶中墨客卿。', '渠') 	prediction:  [('君', 0.5099825263023376), ('谁', 0.0938527062535286)]
('平生事靖退，墨__自要津。', '头') 	prediction:  [('客', 0.07134437561035156), ('墨', 0.06212985888123512)]
('潮满江津猿鸟啼，荆夫__语飞蛮桨。', '楚') 	prediction:  [('无', 0.17239007353782654), ('不', 0.12687981128692627)]
('维时南风薰，木叶晃__碧。', '繁') 	prediction:  [('空', 0.13676896691322327), ('金', 0.13056747615337372)]
('一去别金__，飞沈失相从。', '匣') 	prediction:  [('闺', 0.10392197966575623), ('门', 0.08899002522230148)]
('谓天子__，感于人神。', '孝') 	prediction:  [('心', 0.07555984705686569), ('命', 0.043520599603652954)]
...

```