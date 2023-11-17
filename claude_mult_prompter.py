import datasets
from random import randint
import requests
import pickle
claude = datasets.load_dataset("Norquinal/claude_multiround_chat_1k")
global names
try:
    names = pickle.load("names.mist")
except:
    names = requests.get("https://raw.githubusercontent.com/dominictarr/random-name/master/first-names.txt").text.split("\r\n")
    file = open("names.mist","wb")
    pickle.dump(names,file)
finally:
    print(names.__len__())
def sharegpt_to_prompter(sample):
    sample = sample["conversations"]
    instruction = "<|system|> You are a friendly and helpful AI assistant responding to a user\'s requests.\n"
    conversation = "<START>\n"
    for i,v in enumerate(sample):
        #print(i,v)
        if v["from"] == "human":
            conversation = conversation +"<|prompter|>"+v["value"] +"\n"
        else:
            conversation = conversation +"<|assistant|>"+v["value"] +"\n"
    return {"text":instruction+conversation}
ds = claude.map(sharegpt_to_prompter)
#print(ds["train"][0])
ds = ds.remove_columns(["id","conversations"])
print(ds["train"][0])
ds.push_to_hub("claude_multiround_1k_prompter",private=True)