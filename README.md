# Motion_Correction
2026 Capsone Project

File:

Dataset/Euler_Quar: Euler Angle 형식의 .fbx 파일을 읽어 Quartanion .npy로 변환
                    -> euler_to_quart - Learning용 1개의 Big data(apply Sliding Window)

                    -> euler_to_quart_indi - 각각의 .fbx를 .npy로 변환하여 Bandai_Dataset_npys로 저장.

MLP_CNN: CNN 2D Model

Transformer (Main): Transformer Model

Noise_to_Warudo: send .fbx->.npy file to Warudo (Using VMC Protocol)
VMC_Protocol: 수정중 (Not Perfect)
open_npy: npy 파일을 Terminal에서 확인가능