import os
from maskrcnn_benchmark.data.datasets.relation_tsv import sort_key_by_val

from maskrcnn_benchmark.data.datasets.utils.load_files import find_file_path_in_yaml
# from .bert import BertModel, BertConfig
from transformers import BertTokenizer, BertModel, BertConfig

import torch
from torch import nn
import torch.nn.functional as F
import os.path as op
import json

class EncoderBase(nn.Module):
  def __init__(self, cfg, bert_config_path, encoder_width):
    super().__init__()

    self.cfg = cfg
    self.tokenizer = init_tokenizer()
    config = BertConfig.from_json_file(bert_config_path)
    self.encoder = BertModel(config)

    jsondict_file = op.join(cfg.DATA_DIR,
                            cfg.MODEL.ROI_RELATION_HEAD.VLTRANSE.CLASSES_FN)

    # object classes
    jsondict = json.load(open(jsondict_file, 'r'))
    self.class_to_ind = jsondict['label_to_idx']
    self.class_to_ind['__background__'] = 0
    self.ind_to_class = {v:k for k, v in self.class_to_ind.items()}
    self.classes = sort_key_by_val(self.class_to_ind)

    # relation classes
    self.relation_to_ind = jsondict['predicate_to_idx']
    self.relation_to_ind['__no_relation__'] = 0
    self.ind_to_relation = {v:k for k, v in self.relation_to_ind.items()}
    self.relations = sort_key_by_val(self.relation_to_ind)

  def forward(self, proposal_pairs):
    
    # Extract text from cfg
    obj_str = [', '.join([self.ind_to_class[pair[0].item()], self.ind_to_class[pair[1].item()]]) for pair in proposal_pairs]
    tokens = self.tokenizer(obj_str,  padding='longest', return_tensors="pt").to(proposal_pairs.device)

    output = self.encoder(tokens.input_ids, return_dict=True)
    features = output.last_hidden_state[:,0,:]
    pass

def init_tokenizer():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenizer.add_tokens('__background__') # add background token
  
  return tokenizer

def build_text_encoder(cfg):
  bert_config_path=cfg.MODEL.ROI_RELATION_HEAD.BERT_CONFIG_PATH
  encoder_width=cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
  
  return EncoderBase(cfg, bert_config_path, encoder_width)
