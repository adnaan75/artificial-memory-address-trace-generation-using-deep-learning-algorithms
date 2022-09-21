# Deep_learning_project

## Get start
```bash
git clone git@github.com:adnaan75/Deep_learning_project.git
cd Deep_learning_project
```
## Generate dataset
```bash
cd data_generator
mkdir train_data
# generate memory trace
bash gen_trace.sh 
# generate reuse distance for the generated trace
bash gen_label.sh >> train_data_label.log 
cd ..
```
## Training Seq2Seq model to learn to reproduce the memory trace
```bash
python train_seq2seq.py
```