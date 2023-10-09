
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pdb

# It may require to have proper version of tensorflow-intel.  Uninstall tensorflow-intel and reinstall with proper version.
import tensorflow_text as tf_text

os.chdir("C:/Users/knkas/Desktop/NLP_example")

#---- Using tensorflow_text tokenizer --------------
# https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer
tokenizer = tf_text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())


#---- Using transformer tokenizer ------------------
from transformers import BertJapaneseTokenizer, AutoTokenizer, AutoModel

model_name = "cl-tohoku/bert-base-japanese-v2"
text = "楽しくリズム感覚が身につく"

# May need to pip install "fugashi" "unidic_lite".  
tokenizer = BertJapaneseTokenizer.from_pretrained(
        model_name, 
        #do_subword_tokenize=False,
        #mecab_kwargs={"mecab_dic": None, "mecab_option": "-d, 'C:\mecab-unidic-neologd'},"
        #padding=True, 
        #truncation=True,
    )

# tokenizer.encodeでテキストをトークンIDに,return_tensors='tf'でtensorflow型のテンソルに変換
ids = tokenizer.encode( text, return_tensors='tf')[0]
wakati = tokenizer.convert_ids_to_tokens(ids) #どのようにトークナイズされたか分かち書きで確認
print(ids)
print(wakati)

# AutoTokenizer produced the same result.
tokenizer = AutoTokenizer.from_pretrained(model_name)
ids = tokenizer.encode( text, return_tensors='tf')[0]
wakati = tokenizer.convert_ids_to_tokens(ids) #どのようにトークナイズされたか分かち書きで確認
print(ids)
print(wakati)

# tokenizer will output the dictionary. Note that shape is different.
ids = tokenizer( text, return_tensors="tf" )['input_ids']
tokens = tokenizer( text, return_tensors="tf" )
print(ids)

# Use the pre-trained model.
model = AutoModel.from_pretrained(model_name)

pdb.set_trace()





