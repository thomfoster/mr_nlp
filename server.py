from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from models import Bert, Classifier, Summarizer
from utils import collate_fn
from agent import GeneiAgent

import torch

# Initialize BERT and fine-tune model and wrap together
language_model = Bert(temp_dir='./temp' , load_pretrained_bert=True, bert_config=None)
finetune_model = Classifier(hidden_size=768)
model = Summarizer(language_model, finetune_model)
model.eval()

genei = GeneiAgent(model=model, optimizer=None)
chkpt = torch.load('./checkpoints/chkpt_step_67848mcc_0.319.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(chkpt['state_dict'])

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("plain.html")

@app.route("/_listener", methods=["POST"])
def _listener():
    filepath = request.form.get("filepath", 'Empty filepath', type=str)
    id = request.form.get("id", 0, type=int)
    batch = torch.load('./data/'+filepath)[id]
    batch = collate_fn([batch])
    result = batch.src_txt[0]
    output = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0]
    output = output.data.numpy()[0,:].tolist()
    return jsonify(result=result, output=output)