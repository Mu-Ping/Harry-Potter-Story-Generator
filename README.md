# Harry-Potter-Story-Generator 
訓練一款可以產生哈利波特相關故事的模型，使用「範例故事」當作訓練資料，並根據「預設故事」預測一篇故事。  
此專案於Google colab上開發，檔案結構如下：
|檔案/資料夾|簡述|
|-----|--------|
|HarryPotter-en|內含七篇範例故事       |
|story1.txt  |預設故事一   |
|story2.txt  |預設故事二   |
|story3.txt  |預設故事三   |

⚠️但Text Generation技術困難，不可能產生完美的故事，故此專案專注於兩大重點：
  * 「**詞向量轉換-Word2Vec**」
  
  * 「**嵌入層-Embedding**」。  
## 兩大重點
### **1. 詞向量轉換-Word2Vec**
在使用 RNN (Recurrent Neural Network) 做文字相關的處理時，我們可以利用 gensim 的 Word2Vec 將一個詞彙轉成一個向量表達，即詞向量。

### **2. 嵌入層-Embedding**
如何使用Word2Vec所產生的「詞向量」呢？ 一個簡單的作法是，將 training data 和 testing data 資料裡的那些詞彙以對應的向量取代，然而這樣會很佔記憶體，為了避免這個問題，我們會將 Word2Vec 中的詞彙做個編號(index)，接著將 training / testing data 中的資料以這個編號取代，在 keras 上訓練時，我們則需要將 index 到 vector 的對應傳入Embedding，即可「動態」將詞轉換成對應詞向量。

程式流程如下：  

## A. import 相關套件
```python
import nltk
import numpy as np
import keras
from keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec
from tensorflow.keras import models
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributed, Dropout, GRU
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
nltk.download('punkt');

from google.colab import drive
drive.mount('/content/drive')
```
## B. 資料前處理 
主要有兩個步驟：
### 1. **資料斷詞**  
  將七篇範例故事讀入，使用split("\n") 將文章分句後，再「一句一句」將其斷詞( tokenize )，此處使用nltk.word_tokenize()完成。  
    
### 2. **Word2Vec訓練**   
  將斷詞後的句子使用gensim所提供的Word2Vec方法，訓練詞向量，相關參數如下：  
* word2vec： 訓練資料，格式為[[tokenized句子1], [tokenized句子2], [tokenized句子3]]
* min_count： 「詞」的最小出現頻率，例如：min_count=2表示出現2次以下的詞不會被考慮
* size： window size，即詞向量維度
```python
filename_list=["Harry-Potter-and-the-Chamber-of-Secrets.txt",
        "Harry-Potter-and-the-Deathly-Hallows.txt",
        "Harry-Potter-and-the-Goblet-of-Fire.txt",
        "Harry-Potter-and-the-Half-Blood-Prince.txt",
        "Harry-Potter-and-the-Order-of-the-Phoenix.txt",
        "Harry-Potter-and-the-Philosophers-Stone.txt",
        "Harry-Potter-and-the-Prisoner-of-Azkaban.txt"]
word2vec=[]

#為個別txt檔之word_tokenize
all_txt_word_tokenize=[] 

for file in filename_list:
  single_txt_word_tokenize=[]
  with open("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/HarryPotter-en/"+file, 'r',encoding="utf-8") as f:        
    for i in f.read().split("\n"):                    #建議使用\n分割，整篇文章下去可能會出現錯誤，下方有範例 
      word_tokenize=nltk.word_tokenize(i)           #為甚麼不能用split("")而用word_tokenize 解：https://reurl.cc/Q32y1p
      single_txt_word_tokenize.extend(word_tokenize)
      word2vec.append(word_tokenize) 
  all_txt_word_tokenize.append(single_txt_word_tokenize)
  #word2vec 使用 Cosine Similarity 來計算兩個詞的相似性．這是一個 -1 到 1 的數值，如果兩個詞完全一樣就是 1
  # print(embedding_model.wv.most_similar('one')) #印出Cosine Similarity

embedding_model = Word2Vec(word2vec, min_count=1, size=5)
```
### 補充問題    
* Q1： 為什麼tokenize前要先使用split("\n")分句，直接使用整篇文章來tokenize不行嗎？  
  A1： 因為 " \n " 沒有去除，可能會造成斷詞錯誤，以下為測試：
  ```python
  # 使用整篇文章tokenize；每句使用\n分割再tokenize 差別比較
  with open("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/HarryPotter-en/"+filename_list[1], 'r',encoding="utf-8") as f:
    word_tokenize_1=nltk.word_tokenize(f.read())

  single_txt_word_tokenize=[]
  with open("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/HarryPotter-en/"+filename_list[1], 'r',encoding="utf-8") as f:
    for i in f.read().split("\n"): #為啥不能整個文章下去tokenize
      word_tokenize_2=nltk.word_tokenize(i)
      single_txt_word_tokenize.extend(word_tokenize_2)

  print(word_tokenize_1[670:675])
  print(single_txt_word_tokenize[670:675])

  => ['nearly', 'late.', '”', 'The', 'speaker'] #未使用split("\n")斷句, "late." 斷詞錯誤!!
  => ['nearly', 'late', '.', '”', 'The']
  ```
* Q2： 為甚麼不可以使用spilt(" ")來斷詞，而要使用 nltk   
  A2： 參考：https://reurl.cc/Q32y1p 
## C. 生成對照表
Word2vec的生成結果會有三種對應
  * index to word ( index2word ) ex: "詞的index" -> "詞"  
  * word to index ( word2index ) ex: "詞" -> "詞的index"  
  * index to vector ( index2vector ) ex: "詞的index" -> "詞向量" 
 
我們需要算出三種轉換表 **index2word**、**word2index**、**index2vector**
* **index2word**---=> ["字詞0", "字詞1", "字詞2"]

  * 用來生成word2index，使用[Word2vec_model].wv.index2word算出
* **word2index**---=> {"字詞0": 0, "字詞1": 1, "字詞2": 2}  

  * 用來轉換訓練資料，使用 index2word 算出
* **index2vector**--=> ["字詞0向量", "字詞1向量", "字詞2向量"]  

  * 用來製作Embedding層，使用 [Word2vec_model].wv.vectors可算出
```python
index2word=embedding_model.wv.index2word   
word2index= {}                            
index2vector=embedding_model.wv.vectors   

for i in range(len(index2word)):
  word2index[index2word[i]]=i
```
## D. 模型訓練資料
主要分為以下兩步驟：
### 1. 處理資料
* 為了適當使用資料，不浪費又不過冗，因此採取下圖的結構( 範例：3步長為一筆資料)。
  * training data=[input1, input2, ...]
  * training label=[output1, output2, ...]
  
 <img src="https://i.imgur.com/h0LRsOV.png" width="650" height="200" alt="一張圖片">

* 依照上述結構將斷詞製成training data、training label，本專案採10步長為一筆資料；因為模型預測最終屬於分類問題，output需要轉換成one hot encoding( 在生成器部分實作 )。
  * training data 需依照word2index將input中的"詞"轉換成"詞的index"  
  ( 需與Embedding中的對應關係一致，此處使用word2index )
  
  * training label 需依照word2index將output中的"詞"轉換成"詞的index"  
  ( 也可以自訂不同的對應關係，此處為方便採用word2index對應 )
```python
train_sentence_x=[]
train_sentence_y=[]
step=10 #設定步長

#處理訓練資料------------將資料換成Word2Vec的Index 參考：https://reurl.cc/9XWMVd
for temp in all_txt_word_tokenize:
  x=temp[:-1]
  y=temp[1:]  
  for i in range(0,len(temp)-step, step):   #10步為一筆資料
    tempx=[]
    tempy=[]
    for j in range(step):
      tempx.append(word2index[x[i+j]]) #透過word2index轉換
      tempy.append(word2index[y[i+j]]) #透過word2index轉換
    train_sentence_x.append(tempx)
    train_sentence_y.append(tempy)
train_sentence_x=np.asarray(train_sentence_x)
train_sentence_y=np.asarray(train_sentence_y)
```
### 2. 製作生成器
因為最終模型輸出( output )屬於分類問題( 預測詞的index )，所以需要one hot encoding。共有30000多種詞，單一詞需要30000維的矩陣來處理，如果將訓練資料的output全部替換會造成記憶體負擔
，所以此專案配合keras fit_generator()，使用生成器來產生訓練資料，用多少產生多少，以此減少記憶體負擔。而訓練時的batch size取決於生成器一次回傳的大小，此專案設定為512
```python
allkind=[i for i in range(embedding_model.wv.vectors.shape[0])]
answer=to_categorical(allkind, embedding_model.wv.vectors.shape[0]) #預先存好one hot encoding
batch_size=512  #設定batch_size

def train_sentence_generator():#預防資料過大
  while 1:
    tempx=[]
    tempy=[]
    for i in range(0, len(train_sentence_x)):
      temp=[]
      for j in train_sentence_y[i]:
        temp.append(answer[j])
      tempx.append(train_sentence_x[i])
      tempy.append(temp)  
      if((i+1)%batch_size==0):
        yield (np.asarray(tempx), np.asarray(tempy)) #這個元組（生成器的單個輸出）組成了單個的batch
        tempx=[]
        tempy=[]   
  #重要!!! keras的input都為[[data1],[data2]] 就算只有一個data 也要寫成[[data1]] 不可以是 [data]
```
## E. 模型訓練
較重要的部分為Embedding( input_dim, output_dim, weight )，其功能將input的「詞的index」轉換為「詞向量」:
* input_dim ：所有「詞」的種類，此專案為30000多。
* output_dim ：詞向量維度，即Word2Vec的size參數決定。
* weight ： 此參數繼承自Layer層，為一個list，即Word2Vec的詞向量對應表

其餘模型相關部分可以客製化( 網路架構、優化器、 epoch...)，在此不贅述。但核心根據input、output架構如下:  
<img src="https://i.imgur.com/HYhWqHU.png" width="600" height="400" alt="一張圖片">

```python
model = Sequential()
model.add(Embedding(embedding_model.wv.vectors.shape[0], embedding_model.wv.vectors.shape[1], weights=[embedding_model.wv.vectors], input_length=10))
model.add(LSTM(1000, return_sequences=True))
model.add(Dense(embedding_model.wv.vectors.shape[0], activation='softmax'))
model.summary()

=>>
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 10, 10)            300940    
_________________________________________________________________
lstm (LSTM)                  (None, 10, 1000)          4044000   
_________________________________________________________________
dense (Dense)                (None, 10, 30094)         30124094  
=================================================================
Total params: 34,469,034
Trainable params: 34,469,034
Non-trainable params: 0
_________________________________________________________________

#model = models.load_model('/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/model.h5')  #Adam(0.001)

model.compile(loss=keras.losses.categorical_crossentropy, # 設定 Loss 損失函數
              optimizer=Adam(0.01),      # 設定 Optimizer 最佳化方法，此專案學習重要
              metrics=['accuracy'])
checkpoint = ModelCheckpoint("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/model.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='max')
learning_rate_function = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10, min_lr=0.00000001, mode='max')
model.fit_generator(train_sentence_generator(), steps_per_epoch =len(train_sentence_x)/batch_size, epochs=300, verbose=1, callbacks=[checkpoint, learning_rate_function]) #batch size不能太小：https://reurl.cc/14zQvV
```
## F. 預測
1. 將**預設故事**讀入，將其斷詞( tokenize )，並移除姓名( 因為Embedding"不一定"有人名的對應 )

2. 因為預設字串不到10個字，所以將測試資料補 0  
  ( 最好是補padding，但Embedding需加入對應[0 ,0, 0, 0,...]，但此專案為方便直接補0 )

3. 最後開始預測：
    1.  將一句話(A)丟入模型
    2.  產生一句話(B)
    3.  將(B)的最後一個字存入矩陣(C)，即為文章下個字的預測結果
    4.  移除(A)的第一個字，將(B)最後一個字加入(A)的最後，產生新的(A)
    5.  回到第一步，直到(C)產生到一定長度
    6.  (C)最後即為整篇故事    
4. 重複上述步驟直到所有故事預測完成( 此專案有3篇預設故事 )  

```python
model = models.load_model('/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/model.h5')
txt=["story1.txt", "story2.txt", "story3.txt"]

for i in txt:
  temp_test_data=[]
  with open("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/"+i, 'r',encoding="utf-8") as f:
    text=f.read()
    word_tokenize=nltk.word_tokenize(text.replace("\n",""))
    temp_test_data.extend(word_tokenize[2:]) #人名直接去除(此去除法並不精確)

  test_data=[]
  #初始訓練參數，如果story初始未到10個字，補0
  for _ in range(10-len(temp_test_data)):
    test_data.append(0)
  #初始訓練參數，加入預設故事的前幾個字
  for j in temp_test_data:
    test_data.append(word2index[j])

  #開始預測
  text=text.replace("\n","")+" "
  for _ in range(500):
    nextword=model.predict_classes(np.asarray([test_data]))
    nextword=nextword[0][-1]
    test_data.pop(0) #刪除第一個字
    test_data.append(nextword)
    text=text + index2word[nextword]+" "
    
  with open("/content/drive/My Drive/Colab Notebooks/Ping/自然語言處理/106403546_"+i, 'w') as f:
    f.write(text)
```

@專案： [Harry-Potter-Story-Generator](https://github.com/muping0326/Harry-Potter-Story-Generator )  
@author: [Mu-Ping](https://github.com/Mu-Ping)  
@e-mail： k0326jim@gmail.com
