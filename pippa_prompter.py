import datasets
from random import randint
import requests
import pickle
pippa = datasets.load_dataset("PygmalionAI/PIPPA")
global names
try:
    names = pickle.load("names.mist")
except:
    names = requests.get("https://raw.githubusercontent.com/dominictarr/random-name/master/first-names.txt").text.split("\r\n")
    file = open("names.mist","wb")
    pickle.dump(names,file)
finally:
    print(names.__len__())

def change_to_prompter(sample):
    user_name = names[randint(0,len(names)-1)]
    char_name = sample["bot_name"]
    char_defs:str = sample["bot_definitions"].replace("{{char}}",char_name).replace("{{user}}",user_name)
    char_desc:str = sample["bot_description"].replace("{{char}}",char_name).replace("{{user}}",user_name)
    instuction = f"<|system|>{char_name}'s persona\n{char_desc}\n<START>\n{char_defs}\n\n"
    conversation:str = "<START>\n"
    for i,v in enumerate(sample["conversation"]["message"]):
        if sample["conversation"]["is_human"][i] == True:
            conversation = conversation + f"<|prompter|>{user_name}:{v.replace('{{char}}',char_name).replace('{{user}}',user_name)}<\s>\n"
        else:
            conversation = conversation + f"<|assistant|>{char_name}:{v.replace('{{char}}',char_name).replace('{{user}}',user_name)}\n"
    return {"text":instuction + conversation}
        
ds = pippa.map(change_to_prompter)
print(ds)
ds = ds.remove_columns(["submission_timestamp","categories","bot_id","bot_name","bot_greeting","bot_definitions","bot_description","conversation"])
print(ds)
ds.push_to_hub("PIPPA_prompter",private=True)