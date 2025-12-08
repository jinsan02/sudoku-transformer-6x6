# src/model/transformer.py (6x6 최종 고성능 버전)
import torch
import torch.nn as nn

class SudokuTransformer(nn.Module):
    def __init__(
        self, 
        num_classes=7,    # 0~6 (6x6 스도쿠)
        seq_len=36,       # 6x6 = 36칸
        d_model=256,      # [UP] 뇌 용량 2배 (128 -> 256)
        nhead=8,          # [UP] 시야 2배 (4 -> 8)
        num_layers=8,     # [UP] 사고 깊이 2배 (4 -> 8)
        dropout=0.1
    ):
        super().__init__()
        
        # 1. 숫자 임베딩
        self.token_embedding = nn.Embedding(num_classes, d_model)
        
        # [핵심] 2. 6x6 전용 구조적 위치 임베딩 (행, 열, 박스 정보 추가)
        self.row_embedding = nn.Embedding(6, d_model)
        self.col_embedding = nn.Embedding(6, d_model)
        self.box_embedding = nn.Embedding(6, d_model)
        
        # 3. 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

        # 위치 인덱스 미리 계산 (속도 최적화)
        # 0~35 인덱스를 행/열/박스 번호로 변환
        self.register_buffer('row_idx', torch.arange(seq_len) // 6)
        self.register_buffer('col_idx', torch.arange(seq_len) % 6)
        # 6x6 박스 공식: (행 // 2) * 2 + (열 // 3)
        self.register_buffer('box_idx', (torch.arange(seq_len) // 12) * 2 + (torch.arange(seq_len) % 6) // 3)

    def forward(self, x):
        # 입력이 (Batch, 6, 6)이면 (Batch, 36)으로 펴주기
        if x.dim() == 3:
             x = x.view(x.size(0), -1)
        
        x_emb = self.token_embedding(x)
        
        # 구조적 위치 정보 더하기 (행+열+박스)
        pos_info = (self.row_embedding(self.row_idx) + 
                    self.col_embedding(self.col_idx) + 
                    self.box_embedding(self.box_idx))
        
        x = x_emb + pos_info
        
        x = self.transformer_encoder(x)
        logits = self.output_layer(x)
        
        return logits