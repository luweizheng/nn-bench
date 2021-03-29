import torch
import random

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout, seed=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        # self.dropout = nn.Dropout(dropout)

        self.seed = seed
        self.prob = dropout

    def forward(self, src):
        embedded = self.embedding(src)

        # if self.training:
        #     embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)

        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, dropout, seed=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size)
        self.fc_out = nn.Linear(embed_size + hidden_size * 2, output_size)
        # self.dropout = nn.Dropout(dropout)

        self.seed = seed
        self.prob = dropout

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)

        # if self.training:
        #     embedded = self.dropout(embedded)

        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(output)

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]

        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
