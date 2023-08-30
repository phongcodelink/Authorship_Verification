from pathlib import Path


class BaseConfig:
    """
    BaseConfig class with predefined values
    """

    def __init__(self):
        # Model configurations
        self.model_name = "Author_Verification"
        self.save_top_k = 1
        self.num_workers = 10
        self.n_epochs = 100
        self.batch_size = 1
        self.max_len = 700
        self.lr = 2e-5
        self.n_filters = 128
        self.filter_sizes = [1, 2, 3]
        self.dropout = 0.25
        self.embedding_dim = 50

        # Paths configurations
        base_path = Path(__file__).parents[2].__str__()
        self.raw_data_dir = base_path + "/data/AV_processed_v5"
        self.processed_data_dir = base_path + "/data/Processed"
        self.assets_dir = base_path + "/assets/"
        self.saved_model_path = base_path + "/assets/saved_models/"
        self.language_model_path = "google/flan-t5-base"
        self.csv_logger_path = base_path + "/assets"
        self.best_model_path = self.assets_dir + "saved_models/Author_Verification/version_6/checkpoints/QTag-epoch=02-val_acc=1.00.ckpt"
        self.pair_data = "train_pairs.jsonl"
        self.truth_data = "train_truth.jsonl"
        self.real_pair_data = "real_pairs.jsonl"
        self.real_truth_data = "real_truth.jsonl"
        self.test_pair_data = "test_pairs.jsonl"
        self.test_truth_data = "test_truth.jsonl"
        self.hidden_test_pair_data = "hidden_test_pairs.jsonl"
        self.hidden_test_truth_data = "hidden_test_truth.jsonl"
        self.hidden_2_test_pair_data = "hidden_2_test_pairs.jsonl"
        self.hidden_2_test_truth_data = "hidden_2_test_truth.jsonl"
        self.half_hidden_test_pair_data = "half_hidden_test_pairs.jsonl"
        self.half_hidden_test_truth_data = "half_hidden_test_truth.jsonl"
        self.features_file = "features.pkl"
        self.target2index_file = "target2index.json"
        self.index2target_file = "index2target.json"
        self.pos2index_file = "pos2index.json"
        self.index2pos_file = "index2pos.json"
        self.best_model_path_file = base_path + "/assets/saved_models/Author_Verification/best_model_path.json"


# Example usage:
# config = BaseConfig()
# print(config.model_name)
