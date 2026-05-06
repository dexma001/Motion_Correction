# Motion_Correction
2026 Capsone Project

File:

Dataset/Euler_Quar: Euler Angle 형식의 .fbx 파일을 읽어 Quartanion .npy로 변환

                    - euler_to_quart: Learning용 1개의 Big data(apply Sliding Window)

                    - euler_to_quart_indi: 각각의 .fbx를 .npy로 변환하여 Bandai_Dataset_npys로 저장.

                    - euler_to_quart_advanced: euler->quartenion logic 수정

                    - euler_to_quart_advanced_indi: 위와 동일 / 개별파일

                    - euler_to_quart_indi_raw: 위와 동일 / 개별파일 / Sliding Window 적용 X

MLP_CNN: CNN 2D Model

Transformer (Main): Transformer Model

Send_to_Warudo: send .fbx->.npy file to Warudo (Raw file)
Send_to_Warudo_with_Noise: send .fbx->.npy file to Warudo (with Gaussian Noise)
open_npy: npy 파일을 Terminal에서 확인가능

Miscellaneous: 분류 X