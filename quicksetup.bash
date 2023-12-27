#!/bin/bash 
read -sp 'Input hf token: ' hf_token
read -p 'input yml: ' yml_file
echo 'use deepspeed:'
read deepspeed
read -sp 'wandb token: ' wandbtoken

huggingface-cli login --token $hf_token
wandb login $wandbtoken
if [[$deepspeed -eq 1]]
then
  echo 'running with deepspeed'
  accelerate launch -m axolotl.cli.train $yml_file --deepspeed deepspeed/zero1.json
else
  echo 'running single GPU'
   accelerate launch -m axolotl.cli.train $yml_file

