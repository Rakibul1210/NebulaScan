import os
import logging
import json
import numpy as np
from tqdm import tqdm
from time import time

from collections import Counter, defaultdict
from pandas import json_normalize, concat, DataFrame, merge

import sentencepiece as spm

from constants import *

class JSONTokenizerBPE():
    def __init__(self,
                vocab_size,
                seq_len,
                model_path=None,
                vocab=None,
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=SPEAKEASY_TOKEN_STOPWORDS,
        ):
        super().__init__(
            seq_len,
            vocab_size,
            cleanup_symbols,
            stopwords
        )

        if model_path is not None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=model_path.replace(".model","")+".model")
            logging.info(" [!] Successfully loaded pre-trained tokenizer model!")
            self.model_path = model_path
            self.load_vocab(vocab=vocab)
        else:
            self.tokenizer = spm.SentencePieceTrainer
            msg = " [!] Initialized tokenizer without pre-trained model.\n\t"
            msg += "You need to train tokenizer with .train() or specify 'model_path=' during initialization!"
            logging.warning(msg)
    
    def split_string_to_chunks(self, s, chunkSize=4192):
        """This function should split a long string into smaller chunks of size chunkSize, 
        but it shouldn't split the string in the middle of a word.

        Args:
            s (str): Longstring
            chunkSize (int, optional): _description_. Defaults to 512.

        Returns:
            list: List of smaller strings
        """
        chunks = []
        words = s.split(" ")
        currentChunk = ""
        for word in words:
            if len(currentChunk) + len(word) < chunkSize:
                currentChunk += word + " "
            else:
                chunks.append(currentChunk)
                currentChunk = word + " "
        chunks.append(currentChunk)
        return chunks

    def load_vocab(self, vocab=None):
        if isinstance(vocab, dict):
            self.vocab = vocab
            self.reverse_vocab = {v:k for k,v in self.vocab.items()}
            return

        # parsing default sentencepiece vocab file
        if vocab is None:
            vocab = self.model_path.replace(".model","")+"_vocab.json"
        if not os.path.exists(vocab): # default sentencepiece -- after training
            vocab = self.model_path.replace(".model", "")+".vocab"
        if not os.path.exists(vocab):
            logging.error(f" [!] Vocab file {vocab} does not exist! .load_vocab() failed!")
            return

        with open(vocab, encoding="utf-8") as f:
            if vocab.endswith(".json"):
                self.vocab = json.load(f)
            else:
                data = f.read()
                vocab = [x.split("\t")[0] for x in data.split("\n")]
                self.vocab = {k:i for i,k in enumerate(vocab)}
        # update vocab with special tokens, but ensure that they are unique & at correct locations
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        for k,v in self.special_tokens.items():
            keys[v] = k
        
        self.vocab = dict(zip(keys, values))
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        logging.info(f" [!] Loaded vocab from {vocab}")
        
    def dump_vocab(self):
        vocabFileName = self.model_path.replace(".model","") + "_vocab.json"
        with open(vocabFileName, "w") as f:
            json.dump(self.vocab, f, indent=4)

    def train(
        self,
        jsonData,
        vocab_size=None,
        model_prefix="bpe",
        model_type="bpe",
        split_by_number=False,
        spLength=4192,
        removeTrainFiles=True
    ):
        """
        Trains the tokenizer on the given json data.
        """
        logging.warning(" [*] Data preparation for SentencePiece tokenizer...")
        jsonDataClean = self.clear_json_event(jsonData)
        # splitting a string into chunks of 4192 characters since this sentencepiece limitation
        jsonDataChunks = self.split_string_to_chunks(jsonDataClean.replace("\\\\", "\\"), chunkSize=spLength)
        # dump jsonDataClean to file
        logging.warning(" [*] Saving to disk...")
        trainFile = f"{model_prefix}_trainset_{int(time())}.txt"
        with open(trainFile, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonDataChunks))

        if vocab_size:
            self.vocab_size = vocab_size
        
        trainCmd = " ".join([
            f"--input={trainFile}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={model_type}",
            f"--split_by_number={split_by_number}",
            f"--max_sentence_length={spLength}",
            f"--max_sentencepiece_length=64"
        ])
        logging.warning(f" [!] Training tokenizer with command: {trainCmd}")
        self.tokenizer.Train(trainCmd)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        
        self.model_path = model_prefix
        self.load_vocab()
        self.dump_vocab()

        if removeTrainFiles:
            os.remove(trainFile)
            os.remove(f"{model_prefix}.vocab")
    
    def tokenize(self, inputs):
        """
        Tokenizes the given json data.
        """
        if isinstance(inputs, (str, bytes, dict)) or \
            (isinstance(inputs, list) and isinstance(inputs[0], (str, bytes))):
            inputs = [inputs]
        data_clean = [self.clear_json_event(x) for x in inputs]
        return [self.tokenizer.encode_as_pieces(x) for x in data_clean]

    def encode(self, inputs, pad=True, tokenize=True):
        if not tokenize:
            raise NotImplementedError("SentencePiece tokenizer does not support encode without tokenize!")

        # if single sample, wrap in list
        if isinstance(inputs, (str, bytes, dict)) or \
            (isinstance(inputs, list) and isinstance(inputs[0], (str, bytes))):
            inputs = [inputs]

        data_clean = [self.clear_json_event(x) for x in inputs]
        encoded = [self.tokenizer.encode_as_ids(x) for x in data_clean]
        if pad:
            return self.pad_sequence_list(encoded)
        else:
            return encoded
