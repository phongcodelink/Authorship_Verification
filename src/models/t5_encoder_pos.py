# pylint: disable-msg=no-member
# pylint: disable=too-many-ancestors
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
"""
    AV Project:
        models:
            mt5 encoder finetune
"""
import pytorch_lightning as pl
# ============================ Third Party libs ============================
import torch
import torch.nn.functional as function
import torchmetrics
from torch import nn
# ============================ My packages ============================
from transformers import T5EncoderModel

from .attention import ScaledDotProductAttention


class Classifier(pl.LightningModule):
    """
        Classifier
    """

    def __init__(self, num_classes, t5_model_path, lr, max_len, **kwargs):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.f_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.max_len = max_len
        self.learning_rare = lr

        self.model = T5EncoderModel.from_pretrained(t5_model_path)
        # self.punc_embeddings = nn.Embedding(kwargs["vocab_size"],
        #                                     embedding_dim=kwargs["embedding_dim"],
        #                                     padding_idx=kwargs["pad_idx"])
        # self.punc_embeddings.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs["n_filters"],
                      kernel_size=(fs, self.model.config.d_model))  # kwargs["embedding_dim"]))
            for fs in kwargs["filter_sizes"]
        ])

        self.classifier = nn.Linear(3 * self.model.config.d_model + (len(kwargs["filter_sizes"]) * kwargs["n_filters"]),
                                    num_classes)
        self.attention = ScaledDotProductAttention(3 * self.model.config.d_model)

        self.max_pool = nn.MaxPool1d(max_len)
        self.max_pool_info = nn.MaxPool1d(max_len // 4)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        punctuation = self.model(batch["punctuation"]).last_hidden_state  # .permute(0, 2, 1)

        punctuation = punctuation.unsqueeze(1)
        # # embedded_cnn = [batch_size, 1, sent_len, emb_dim]
        #
        punctuation = [torch.nn.ReLU()(conv(punctuation)).squeeze(3) for conv in self.convs]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]
        #
        punctuation = [function.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in punctuation]
        # # pooled_n = [batch_size, n_filters]
        #
        punctuation = torch.cat(punctuation, dim=1)
        # cat_cnn = [batch_size, n_filters * len(filter_sizes)]

        output_encoder = self.model(batch["input_ids"]).last_hidden_state  # .permute(0, 2, 1)
        information = self.model(batch["information"]).last_hidden_state  # .permute(0, 2, 1)
        pos = self.model(batch["pos"]).last_hidden_state  # .permute(0, 2, 1)

        features = torch.cat((output_encoder, information, pos), dim=2)

        context, attn = self.attention(features, features, features)
        output = context.permute(0, 2, 1)
        features = function.max_pool1d(output, output.shape[2]).squeeze(2)
        features = torch.cat((features, punctuation), dim=1)

        final_output = self.classifier(features)
        return final_output

    # def forward(self, batch):
    #     punctuation = self.model(batch["punctuation"]).last_hidden_state#.permute(0, 2, 1)
    #
    #     # punctuation = self.punc_embeddings(punctuation)
    #     # punctuation = [batch_size, sent_len, emb_dim]
    #
    #     punctuation = punctuation.unsqueeze(1)
    #     # # embedded_cnn = [batch_size, 1, sent_len, emb_dim]
    #     #
    #     punctuation = [torch.nn.ReLU()(conv(punctuation)).squeeze(3) for conv in self.convs]
    #     # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]
    #     #
    #     punctuation = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in punctuation]
    #     # # pooled_n = [batch_size, n_filters]
    #     #
    #     punctuation = torch.cat(punctuation, dim=1)
    #     # cat_cnn = [batch_size, n_filters * len(filter_sizes)]
    #
    #     output_encoder = self.model(batch["input_ids"]).last_hidden_state.permute(0, 2, 1)
    #     information = self.model(batch["information"]).last_hidden_state.permute(0, 2, 1)
    #     pos = self.model(batch["pos"]).last_hidden_state.permute(0, 2, 1)
    #
    #     pos_pool = self.max_pool(pos).squeeze(2)
    #     maxed_pool = self.max_pool(output_encoder).squeeze(2)
    #     information = self.max_pool_info(information).squeeze(2)
    #     # punctuation = self.max_pool(punctuation).squeeze(2)
    #     features = torch.cat((punctuation, maxed_pool, pos_pool), dim=1)
    #
    #     final_output = self.classifier(features)
    #     return final_output

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'train_loss': loss,
                        'train_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'train_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'train_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'train_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'predictions': outputs, 'labels': label}

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'val_loss': loss,
                        'val_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'val_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'val_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'val_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'test_loss': loss,
                        'test_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'test_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'test_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'test_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """

        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]