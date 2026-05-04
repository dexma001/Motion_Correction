import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
from tqdm import tqdm
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Length, d_model)
        return x + self.pe[:, :x.size(1)]

class MotionCorrectionTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Input -> Transformer Layer mapping
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # 2 Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. PyTorch Transformer Encoder 레이어 사용
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Transformer -> Output Layer mapping
        self.output_proj = nn.Linear(d_model, feature_dim)

    def forward(self, src):
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        delta = self.output_proj(x) #Correction Value
        
        # 원본 모션에 보정값을 더해 최종 모션을 생성 (Residual Connection)
        return src + delta
   
class MotionDataset(Dataset):
    def __init__(self, npy_filepath):
        print(f"데이터셋 로딩 중: {npy_filepath}")
        
        # Data load
        self.data = np.load(npy_filepath)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        print(f"로딩 완료! 데이터 형태: {self.data.shape}")

    def __len__(self):
        # 총 윈도우(샘플) 개수 반환 (예: 242604)
        return len(self.data)

    def __getitem__(self, idx):
        # 1개의 윈도우(60프레임) 반환
        return self.data[idx]
 
def main():
    dataset_path = "Bandai_Dataset_Unity.npy"
    
    if not os.path.exists(dataset_path):
        print(f"에러: {dataset_path} 파일을 찾을 수 없습니다.")
        return
    
    dataset = MotionDataset(dataset_path)
    num_samples, seq_length, feature_dim = dataset.data.shape
    print(f"=> 시퀀스 길이: {seq_length}, 차원 수: {feature_dim} 자동 설정됨")
    
    # 설정값
    batch_size = 128
    
    # Original: num_workers = 4 / Window는 num_workers = 0 권장
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionCorrectionTransformer(feature_dim=feature_dim).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- 학습 루프 ---
    epochs = 1
    noise_level = 0.05
    
    log_file_path = "loss_log.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Training Loss Log ===\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, clean_batch in enumerate(dataloader):
            y_clean = clean_batch.to(device)
            
            noise = torch.randn_like(y_clean) * noise_level
            
            '''
            # Frame Skip
            freeze_mask = (torch.rand(y_clean.shape[0], y_clean.shape[1], 1, device=device) > 0.10).float()
            '''
            
            x_noisy = (y_clean + noise) #*freeze_mask
            
            
            #x_noisy = y_clean.clone()
            output = model(x_noisy)
            loss = criterion(output, y_clean)
            
            #Loss
            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                log_text = f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                print(log_text)
                
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(log_text + "\n")
        
        avg_loss = total_loss / len(dataloader)
        epoch_summary = f"✅ Epoch [{epoch+1}/{epochs}] 완료 - Average Loss: {avg_loss:.6f}"
        print(epoch_summary)
            
        with open(log_file_path, "a", encoding="utf-8") as f:
             f.write(epoch_summary + "\n\n")
        
        
    print("\n=== 학습 완료! Parameter(가중치) 분석 ===")

    # 4. 학습된 Parameter 값 확인
    layer_count = len(list(model.named_parameters()))
    print(f"총 레이어 수: {layer_count}개")
    
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"  -> Shape: {list(param.data.shape)}")
            print(f"  -> Mean weight: {param.data.mean().item():.6f}")
            print(f"  -> Max weight:  {param.data.max().item():.6f}")
            print("-" * 40)
    '''    
        
    # 5. 모델 가중치(Weight) 파일로 저장
    save_path = "motion_correction_Transformer_noiselevel_0.05.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n성공: 모델 가중치가 '{save_path}'로 안전하게 저장되었습니다!")    
    
if __name__ == "__main__":
    main()