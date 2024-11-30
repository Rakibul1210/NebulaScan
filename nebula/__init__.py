import nebula
from nebula.preprocessing import (
    JSONTokenizerBPE,
    PEDynamicFeatureExtractor,
)
from nebula.preprocessing.pe import is_pe_file
from nebula.models import Cnn1DLinearLM, MLP
from nebula.models import TransformerEncoderChunks

import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from typing import Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Linear
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


class Nebula:
    def __init__(
            self,
            vocab_size: int = 50000,
            seq_len: int = 512,
            tokenizer: str = None,
            speakeasy_config: str = None,
            # pre-trained files
            vocab_file: str = None,
            bpe_model_file: str = None,
            torch_model_file: str = None,
            torch_model_config: dict = None,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # input checks
        assert tokenizer in ['bpe', 'whitespace'], "tokenizer must be in ['bpe', 'whitespace']"
        assert vocab_file is None or os.path.exists(vocab_file), f"[-] {vocab_file} doesn't exist..."
        assert bpe_model_file is None or os.path.exists(bpe_model_file), f"[-] {bpe_model_file} doesn't exist..."
        assert torch_model_file is None or os.path.exists(torch_model_file), f"[-] {torch_model_file} doesn't exist..."
        assert speakeasy_config is None or os.path.exists(speakeasy_config), f"[-] {speakeasy_config} doesn't exist..."

        # dynamic extractor setup
        self.dynamic_extractor = PEDynamicFeatureExtractor(speakeasyConfig=speakeasy_config) 

        # config
        if self.vocab_size == 50000:
            if tokenizer == "bpe":
                if bpe_model_file is None:
                    bpe_model_file = os.path.join(os.path.dirname(nebula.__file__), "objects", "bpe_50000_sentencepiece.model")
                if vocab_file is None:
                    vocab_file = os.path.join(os.path.dirname(nebula.__file__), "objects", "bpe_50000_vocab.json")
                if torch_model_file is None:
                    torch_model_file = os.path.join(os.path.dirname(nebula.__file__), "objects", "bpe_50000_torch.model")
        else:
            msg = f"[-] Nebula supports pre-trained models only for vocab_size = 50000, you have: {self.vocab_size}"
            msg += " | No pre-trained objects loaded! Be sure to train tokenizer and PyTorch model!"
            logging.warning(msg)

        # vocab size adjustment
        if vocab_file is not None:
            with open(vocab_file) as f:
                # actual vocab size might differ slightly because of special tokens
                nebula_vocab = json.load(f)
            self.vocab_size = len(nebula_vocab)
        
        # tokenizer initialization
        if tokenizer == 'bpe':
            self.tokenizer = JSONTokenizerBPE(
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                vocab=vocab_file,
                model_path=bpe_model_file
            ) 
        logging.info(" [!] Tokenizer ready!")
        
        # PyTorch model initialization
        if torch_model_config is None:
            torch_model_config = {
                "vocab_size": self.vocab_size,
                "maxlen": self.seq_len,
                "chunk_size": 64, # self-attention window size
                "dModel": 64,  # embedding & transformer dimension
                "nHeads": 8,  # number of heads in nn.MultiheadAttention
                "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
                "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                "numClasses": 1, # binary classification
                "classifier_head": [64], # classifier head depth
                "layerNorm": False,
                "dropout": 0.3,
                "norm_first": True,
                "pooling": "flatten"
            }
        self.model = TransformerEncoderChunks(**torch_model_config)

        state_dict = torch.load(torch_model_file) if torch_model_file is not None else {}
        self.model.load_state_dict(state_dict)
        logging.info(f" [!] Model ready!")

    def dynamic_analysis_pe_file(self, pe_file: Union[str, bytes]) -> dict:
        if isinstance(pe_file, str):
            with open(pe_file, "rb") as f:
                bytez = f.read()
        elif isinstance(pe_file, bytes):
            bytez = pe_file
        else:
            raise ValueError("preprocess(): data must be a path to a PE file or a bytes object")

        dynamic_features_json = self.dynamic_extractor.emulate(data=bytez)
        return dynamic_features_json

    def preprocess(self, emulation_report: dict) -> np.ndarray:
        dynamic_features = self.tokenizer.encode(emulation_report)
        return dynamic_features
    
    def predict_proba(self, dynamic_features: np.ndarray) -> float:
        dynamic_features = torch.Tensor(dynamic_features).long()
        with torch.no_grad():
            logits = self.model(dynamic_features)
        return torch.sigmoid(logits).item()
    
    def predict_sample(self, pe_file: Union[str, bytes]) -> float:
        dynamic_features_json = self.dynamic_analysis_pe_file(pe_file)
        dynamic_features = self.preprocess(dynamic_features_json)
        return self.predict_proba(dynamic_features)



# TODO: ModelTrainer(object):
    