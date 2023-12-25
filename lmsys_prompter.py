import datasets
from random import randint
import requests
import pickle
lmsys = datasets.load_dataset("lmsys/lmsys-chat-1m")
global names
try:
    names = pickle.load("names.mist")
except:
    names = requests.get("https://raw.githubusercontent.com/dominictarr/random-name/master/first-names.txt").text.split("\r\n")
    file = open("names.mist","wb")
    pickle.dump(names,file)
finally:
    print(str(names.__len__())+" Names loaded")

def sharegpt_to_prompter(sample):
    #if sample["model"] != "claude-2" or "claude-1" or "gpt-4" or "gpt-3":
    #    return None
    sample = sample["conversation"]
    instruction = "<|system|> You are a friendly and helpful AI assistant responding to a user\'s requests.\n"
    conversation = "<START>\n"
    for i,v in enumerate(sample):
        #print(i,v)
        if v["role"] == "user":
            conversation = conversation +"<|prompter|>"+v["content"] +"\n"
        else:
            conversation = conversation +"<|assistant|>"+v["content"] +"\n"
    return {"text":instruction+conversation}
ds = lmsys.map(sharegpt_to_prompter)
print(ds["train"][0])

global idx
global unwanted_models 
unwanted_models = []
idx = []
def remove_unwanted_models(sample):
    