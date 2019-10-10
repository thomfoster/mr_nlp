from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from models import Bert, Classifier, Summarizer
from utils import collate_fn
from utils import yang_encode
from agent import GeneiAgent

import torch
from pytorch_transformers import BertTokenizer
import numpy as np

# Initialize BERT and fine-tune models and wrap them together
language_model = Bert(temp_dir='./temp' , load_pretrained_bert=True, bert_config=None)
finetune_model = Classifier(hidden_size=768)
model = Summarizer(language_model, finetune_model)
model.eval()

# Load in our best model so far
chkpt = torch.load('./checkpoints/chkpt_step_67848mcc_0.319.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(chkpt['state_dict'])

# Create a tokenizer for processing our own input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("textbox.html")

@app.route("/_summarise_files", methods=["POST"])
def _summarise_files():
    filepath = request.form.get("filepath", 'Empty filepath', type=str)
    id = request.form.get("id", 0, type=int)

    batch = torch.load('./data/'+filepath)[id]
    batch = collate_fn([batch])
    text = batch.src_txt[0]  # src_txt in batch is 2D

    scores = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0]
    scores = scores.data.numpy()[0,:].tolist()

    return jsonify(text=text, scores=scores)

@app.route("/_summarise_textbox", methods=["POST"])
def _summarise_textbox():
    text = request.form.get("textbox", "No text input", type=str)
    s = yang_encode(tokenizer, text, max_seq_len=512)
    batch = collate_fn([s])

    text = batch.src_txt[0]  # src_txt in batch is 2D

    scores = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0]
    scores = scores.data.numpy()[0,:].tolist()

    return jsonify(text=text, scores=scores)
