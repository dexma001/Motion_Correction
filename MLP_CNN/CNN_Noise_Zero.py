import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import os
from tqdm import tqdm

# 1D CNN 모델 정의
class MotionCNN(nn.Module):
    def __init__(self, in_channels):
        super(MotionCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, in_channels, kernel_size=1) 
        )
        
    def forward(self, x):
        return self.encoder(x)

def main():
    # 1. 데이터 로드 (실제 경로)
    # shape = [242604, 60, 154]
    data = np.load('Bandai_Dataset_Unity.npy')
    data = torch.tensor(data, dtype=torch.float32)

    # (Batch, Length, Channel) -> (Batch, Channel, Length) for Conv1d
    # [242604, 60, 154] -> [242604, 154, 60]
    X_CNN = data.permute(0, 2, 1)
    Y_CNN = X_CNN.clone()

    # 2. 데이터셋 및 로더 준비
    dataset = TensorDataset(X_CNN, Y_CNN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 데이터 양이 많으므로 batch_size를 크게 잡는 것이 빠릅니다.
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # 3. 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionCNN(in_channels=data.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. 학습 루프
    epochs = 1
    noise_level = 0.05
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            '''
            noise = torch.rand(1, device = device).item()* noise_level
            batch_x = batch_x + torch.randn_like(batch_x) * noise
            '''
            
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx+1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            
        avg_loss = total_loss / (batch_idx + 1)
        print(f"✅ Epoch [{epoch+1}/{epochs}] 완료 - Average Loss: {avg_loss:.6f}")
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                '''
                vx = vx + torch.randn_like(vx) * noise_level
                '''
                v_out = model(vx)
                val_loss += criterion(v_out, vy).item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    torch.save(model.state_dict(), "motion_correction_CNN_ZN.pth")
    
if __name__ == "__main__":
    main()