# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        configuration:
                config.py
"""

# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    """
    BaseConfig class is written to write configs in it
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="Author_Verification")

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=1,
                                 help="...")
        self.parser.add_argument("--max_len", type=int, default=700,
                                 help="Maximum length of inputs")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--n_filters", type=int,
                                 default=128,
                                 help="...")
        self.parser.add_argument("--filter_sizes", type=int,
                                 default=[1, 2, 3],
                                 help="...")
        self.parser.add_argument("--dropout", type=float,
                                 default=0.25,
                                 help="...")
        self.parser.add_argument("--embedding_dim", type=int,
                                 default=50,
                                 help="...")

    def add_path(self) -> None:
        """
        function to add path

        Returns:
            None

        """
        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/AV_processed_v4")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/saved_models/"),

        self.parser.add_argument("--language_model_path", type=str,
                                 default="google/flan-t5-base",
                                 help="Path of the multilingual lm model dir")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")
        self.parser.add_argument(
            "--best_model_path", type=str,
            default="/content/drive/Shareddrives/LEAP_hivemind/Fraud_detection/data/one-one_conversations/AV_processed/T5/Authorship_Verification/assets/saved_models/Author_Verification/version_12/checkpoints/QTag-epoch=07-val_acc=1.00.ckpt")
        self.parser.add_argument("--pair_data", type=str, default="train_pairs.jsonl")
        self.parser.add_argument("--truth_data", type=str, default="train_truth.jsonl")
        self.parser.add_argument("--real_pair_data", type=str, default="real_pairs.jsonl")
        self.parser.add_argument("--real_truth_data", type=str, default="real_truth.jsonl")
        self.parser.add_argument("--test_pair_data", type=str, default="test_pairs.jsonl")
        self.parser.add_argument("--test_truth_data", type=str, default="test_truth.jsonl")
        self.parser.add_argument("--hidden_test_pair_data", type=str, default="hidden_test_pairs.jsonl")
        self.parser.add_argument("--hidden_test_truth_data", type=str, default="hidden_test_truth.jsonl")
        self.parser.add_argument("--hidden_2_test_pair_data", type=str, default="hidden_2_test_pairs.jsonl")
        self.parser.add_argument("--hidden_2_test_truth_data", type=str, default="hidden_2_test_truth.jsonl")
        self.parser.add_argument("--half_hidden_test_pair_data", type=str, default="half_hidden_test_pairs.jsonl")
        self.parser.add_argument("--half_hidden_test_truth_data", type=str, default="half_hidden_test_truth.jsonl")

        self.parser.add_argument("--features_file", type=str, default="features.pkl")

        self.parser.add_argument("--target2index_file", type=str, default="target2index.json")
        self.parser.add_argument("--index2target_file", type=str, default="index2target.json")

        self.parser.add_argument("--pos2index_file", type=str, default="pos2index.json")
        self.parser.add_argument("--index2pos_file", type=str, default="index2pos.json")

        self.parser.add_argument("--best_model_path_file", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/saved_models/"
                                                            "Author_Verification/"
                                                            "best_model_path.json")

    def get_config(self):
        """

        :return:
        """
        self.add_path()
        return self.parser.parse_args()
