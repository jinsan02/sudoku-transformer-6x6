# src/model/transformer.py
import torch
import torch.nn as nn

class SudokuTransformer(nn.Module):
    def __init__(
        self, 
        config,  # Config 객체를 통째로 받음
        d_model=256,
        nhead=8,
        num_layers=8,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = config.SEQ_LEN
        self.grid_size = config.GRID_SIZE
        self.box_h = config.BOX_H
        self.box_w = config.BOX_W
        
        # 임베딩
        self.token_embedding = nn.Embedding(config.NUM_CLASSES, d_model)
        self.row_embedding = nn.Embedding(self.grid_size, d_model)
        self.col_embedding = nn.Embedding(self.grid_size, d_model)
        self.box_embedding = nn.Embedding(self.grid_size, d_model)
        
        # 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, config.NUM_CLASSES)

        # 인덱스 버퍼 미리 계산 (일반화된 공식)
        self.register_buffer('row_idx', torch.arange(self.seq_len) // self.grid_size)
        self.register_buffer('col_idx', torch.arange(self.seq_len) % self.grid_size)
        self.register_buffer('box_idx', 
            (torch.arange(self.seq_len) // (self.grid_size * self.box_h)) * self.box_h + 
            (torch.arange(self.seq_len) % self.grid_size) // self.box_w
        )

    def forward(self, x):
        if x.dim() == 3: x = x.view(x.size(0), -1)
        
        x = self.token_embedding(x)
        pos = self.row_embedding(self.row_idx) + \
              self.col_embedding(self.col_idx) + \
              self.box_embedding(self.box_idx)
        
        logits = self.output_layer(self.transformer_encoder(x + pos))
        return logits