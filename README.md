# Motion_Correction

2026 Capsone Project

## 1. File Structure
#### Dataset/Euler_Quar: Euler Angle 형식의 .fbx 파일을 읽어 Quartanion .npy로 변환
- euler_to_quart: Learning용 1개의 Big data(apply Sliding Window)

- euler_to_quart_indi: 각각의 .fbx를 .npy로 변환하여 Bandai_Dataset_npys로 저장.

- euler_to_quart_advanced: euler->quartenion logic 수정

- euler_to_quart_advanced_indi: 위와 동일 / 개별파일

 - euler_to_quart_advanced_indi_raw: 위와 동일 / 개별파일 / Sliding Window 적용 X  

***  
#### MLP_CNN: CNN 2D Model

#### Transformer (Main): Transformer Model  

#### *_weight: training loss log
***  

#### Warudo_Send_Temp: Looking for raw data in Warudo
  
***  
### *Under this line, there is no classification folder*
***  
 
#### Send_to_Warudo: send .fbx->.npy file to Warudo (Raw file)

#### Send_to_Warudo_with_Noise: send .fbx->.npy file to Warudo (with Gaussian Noise)

#### open_npy: npy 파일을 Terminal에서 확인가능  
  
***    
#### Miscellaneous: 분류 X     
  

## 2. Current Work  
- ~26.05.13: epochs = 30 / noise_level = 0.05로 수렴함을 확인함.
    + Related: Transformer_weight
- 26.05.13 ~ : 원본 .bvh file을 .npy 형식으로 변환했을 때(noise X), Warudo 상에서 자연스럽게 보이는지를 확인하고자 함.
    + Related: Warudo_Send_Temp
        + Sample_Data: converted .npy data
        + euler_to_quart_advanced_indi_raw: convert .bvh to .npy  
         (target: dataset-1_dash_active_001.bvh)
        + Send_to_Warudo_with_Noise: **Not related**
        + Send_to_Warudo: Send converted .npy data to Warudo
