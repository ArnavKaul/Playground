#INSPIRATION: BPE algo
import re
class MyTokenizer:
     def __init__(self,vocab_size):
        # self.text=text
        self.vocab_size=vocab_size
        self.str_to_int={}
        self.int_to_str={}


     def train(self,vocab_size,verbose=False): 
          with open("test/data.txt","r") as f:
            content=f.read()
          
          sorted_vocab=sorted(list(set(content)))
          
          for idx, char in enumerate(sorted_vocab):
            self.str_to_int[char] = idx 
            self.int_to_str[idx] = char
            # They are not single integer or string primitive variables. 
            # they are Dictionaries 
    
     

     def encode(self,text):  #converts strings to tokens
         op_token=[]
         for char in text:
            if char in self.str_to_int:
                op_token.append(self.str_to_int[char])
         return op_token
     

     def decode(self,tokens): #convertss  token_ids to string    
         
         op_token_str=[]
         for i in tokens:
            if i in self.int_to_str:
                 op_token_str.append(self.int_to_str[i])
         return op_token_str


tokenizer = MyTokenizer(vocab_size=100)
tokenizer.train(vocab_size=100)
ids = tokenizer.encode("Akbeast")
print("Encoded IDs:", ids)

original_text = tokenizer.decode(ids)
print("Decoded Text:", original_text)

     