# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        src:
            inferencer.py
"""

# ============================ Third Party libs ============================
import logging
import os
import numpy as np
import torch
from transformers import T5Tokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# ============================ My packages ============================
from configuration import BaseConfig
from indexer import TokenIndexer
from data_preparation import prepare_av_data, AVFeatures
from models.t5_model import Classifier
from models import build_checkpoint_callback
from src.dataset import DataModule
from src.indexer import Indexer

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    # ------------------------------ Create CSVLogger Instance -------------------
    LOGGER = CSVLogger(save_dir=ARGS.saved_model_path, name=ARGS.model_name)

    # -------------------------------- Load Data----------------------------------

    FIRST_TEXT, SECOND_TEXT, TARGETS = prepare_av_data(
        pair_data_path=os.path.join(ARGS.raw_data_dir, ARGS.pair_data),
        truth_data_path=os.path.join(ARGS.raw_data_dir, ARGS.truth_data)
    )
    logging.info("test set contain %s sample ...", len(FIRST_TEXT))

    # ------------------------------ Load T5 Tokenizer ---------------------------
    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_path)

    # ----------------------------- Load Model -----------------------------------
    MODEL = Classifier.load_from_checkpoint(ARGS.best_model_path)

    # ------------------------------ Extract Features -----------------------------
    # features[0] --> pos
    # features[1] --> punctuation adn emoji
    # features[2] --> author-specific and topic-specific information
    av_features_obj = AVFeatures(
        datasets=[FIRST_TEXT, SECOND_TEXT],
        tokenizer=word_tokenize,
        pos_tagger=pos_tag)

    # print(FIRST_TEXT[:10])
    # print(SECOND_TEXT[:10])
    # print(TARGETS[:10])

    FIRST_TEXT_FEATURES, SECOND_TEXT_FEATURES = av_features_obj()
    logging.info("Features are extracted.")

    # --------------------------------- Target Indexer ----------------------------------
    TARGET_INDEXER = Indexer(vocabs=TARGETS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.target2index_file),
                        idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2target_file))

    TARGETS_CONVENTIONAL = [[target] for target in TARGETS]
    INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TARGETS_CONVENTIONAL)

    POS_INDEXER = TokenIndexer()
    POS_INDEXER.load(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.pos2index_file),
                     idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2pos_file))

    FIRST_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(FIRST_TEXT_FEATURES[0])
    SECOND_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(SECOND_TEXT_FEATURES[0])

    # ---------------------------- Prepare of aggregated data -------------------------------
    COLUMNS2DATA = {"first_text": FIRST_TEXT[:10],
                    "second_text": SECOND_TEXT[:10],
                    "first_punctuations": FIRST_TEXT_FEATURES[1],
                    "second_punctuations": SECOND_TEXT_FEATURES[1],
                    "first_information": FIRST_TEXT_FEATURES[2],
                    "second_information": SECOND_TEXT_FEATURES[2],
                    "first_pos": FIRST_TEXT_FEATURES_POS,
                    "second_pos": SECOND_TEXT_FEATURES_POS,
                    "targets": INDEXED_TARGET}

    DATA = {"test_data": COLUMNS2DATA}

    # ----------------------------- Create Data Module -------------------------------
    DATA_MODULE = DataModule(data=DATA, config=ARGS, tokenizer=T5_TOKENIZER)
    DATA_MODULE.setup()

    # -------------------------- Instantiate the Model Trainer -----------------------
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_acc", patience=100, mode="max")
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=ARGS.save_top_k,
                                                    monitor="val_acc",
                                                    mode="max",
                                                    filename="QTag-{epoch:02d}-{val_acc:.2f}")
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, devices=1, accelerator="gpu",
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK], logger=LOGGER)

    # ------------------------- Make Prediction -------------------------------------
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)
    # print(PREDICTIONS)
