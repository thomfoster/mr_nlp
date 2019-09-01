import torch.nn as nn
import torch

from pytorch_transformers import BertModel

class Bert(nn.Module):
    "Wrapper for the pretrained Bert module"
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if load_pretrained_bert:
            self.model = BertModel.from_pretrained('bert-base-uncased',cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)
    
    def forward(self, x, segs, mask_attn):
        print('input to bert shape:',x.shape)
        encoded_layers, _ =self.model(x, segs, attention_mask=mask_attn)
        print('encoded layers shape:',encoded_layers.shape)
        final_vec = encoded_layers
        print('bert output shape:',final_vec.shape)
        return final_vec


class Classifier(nn.Module):
    "Simple logistic regression fine tuning layer"
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask_clss):
        print('input going into sigmoid:',x.shape)
        h = self.linear1(x).squeeze(-1) # squeeze(-1) removes last axis
        sent_scores = self.sigmoid(h) * mask_clss
        return sent_scores


class Summarizer(nn.Module):
    "State of the art extractive summarization"
    def __init__(self, language_model, finetune_model):
        
        super(Summarizer, self).__init__()
        self.language_model = language_model
        self.finetune_model = finetune_model
        
        self.to('cuda:0')
        
    def forward(self, x, segs, clss, mask_attn, mask_clss):
        # Pass input into language model
        final_vec = self.language_model(x, segs, mask_attn)
         # Select out only clss vectors
        encoded_clss_tokens = final_vec[torch.arange(final_vec.size(0)).unsqueeze(1).type(torch.long), clss.type(torch.long)]
        
         # For each of the 768 bert indices, apply same value of mask
        encoded_clss_tokens *= mask_clss[:,:,None]
        
        # Put the clss tokens into fine tune layers
        sent_scores = self.finetune_model(encoded_clss_tokens, mask_clss).squeeze(-1)
        
        return sent_scores, mask_clss
