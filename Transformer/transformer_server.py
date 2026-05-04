import torch
from collections import deque
import time

# 1. 모델 로드 및 버퍼 초기화
model = LoadYourTrainedEncoderModel("weights.pth")
model.eval()

window_size = 60
look_ahead = 4 # 4프레임 미래를 보고 보정 (약 66ms 지연)
buffer = deque(maxlen=window_size)

# (처음엔 버퍼를 기본 포즈로 채워둠)
for _ in range(window_size):
    buffer.append(get_default_pose_6d()) 

def on_vmc_message_received(raw_motion_data):
    # 2. VMC 데이터 수신 및 변환
    motion_6d = convert_vmc_to_6d(raw_motion_data)
    
    # 3. 버퍼 업데이트 (가장 오래된 것 빠지고 최신 들어감)
    buffer.append(motion_6d)
    
    # 4. 텐서 변환 및 모델 추론
    with torch.no_grad():
        input_tensor = torch.tensor(list(buffer)).unsqueeze(0) # Shape: (1, 60, Dim)
        output_tensor = model(input_tensor) # Shape: (1, 60, Dim)
    
    # 5. Look-ahead 인덱스에서 1프레임만 추출 (맨 끝에서 look_ahead 만큼 앞의 프레임)
    target_idx = -1 - look_ahead 
    corrected_frame_6d = output_tensor[0, target_idx, :].numpy()
    
    # 6. Unity로 보낼 포맷으로 변환 후 전송
    final_quaternions = convert_6d_to_quaternion(corrected_frame_6d)
    send_vmc_to_warudo(final_quaternions)

# OSC 서버 무한 루프 실행...

'''
import torch
from collections import deque
import numpy as np

# 1. 앞서 정의한 Encoder-Only 모델 클래스 (서버 스크립트에도 선언되어 있어야 함)
# class EncoderOnlyMotionTransformer(nn.Module):
#     ... (생략: 이전 답변의 모델 코드와 동일) ...

def main_server():
    # --- 1. 모델 준비 및 가중치(Weight) 로드 ---
    feature_dim = 132
    window_size = 60
    look_ahead = 4 # 지연 시간(Latency)과 보정 품질 간의 타협점
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("모델 가중치를 불러오는 중...")
    model = EncoderOnlyMotionTransformer(feature_dim=feature_dim).to(device)
    
    # [핵심] 아까 학습해서 저장해둔 가중치를 덮어씌웁니다.
    model.load_state_dict(torch.load("encoder_only_weights.pth", map_location=device))
    
    # [매우 중요] 학습 모드(Dropout 등)를 끄고 실전(추론) 모드로 전환
    model.eval() 

    # --- 2. 슬라이딩 윈도우 버퍼 초기화 ---
    # 초기에는 기본 포즈(0으로 채워진 배열 등)로 60프레임을 꽉 채워둡니다.
    buffer = deque(
        [np.zeros(feature_dim, dtype=np.float32) for _ in range(window_size)], 
        maxlen=window_size
    )

    print("서버 준비 완료! VMC 데이터를 기다립니다...")

    # --- 3. 실시간 OSC 수신 루프 (가상 시뮬레이션) ---
    # 실제로는 python-osc의 dispatcher.map() 콜백 함수 안에 아래 로직이 들어갑니다.
    
    def process_incoming_vmc_frame(new_frame_6d):
        """
        Performer로부터 1프레임의 새로운 데이터가 들어올 때마다 실행되는 함수
        new_frame_6d: shape (132,) 의 numpy 배열
        """
        # 버퍼 맨 뒤에 새 프레임 추가 (가장 오래된 앞쪽 프레임은 자동으로 밀려남)
        buffer.append(new_frame_6d)
        
        # 버퍼 데이터를 PyTorch 텐서로 변환: (1, 60, 132) 형태
        input_tensor = torch.tensor(np.array(buffer)).unsqueeze(0).to(device)
        
        # 모델 추론 (역전파 연산 방지 - 속도 향상 및 메모리 절약)
        with torch.no_grad():
            output_tensor = model(input_tensor) # 타겟(tgt) 입력 없이 하나만 들어감!
            
        # 60프레임의 출력 결과 중, 우리가 쓸 단 '1프레임'만 추출
        # 뒤에서 5번째 프레임 (look_ahead가 4일 경우)
        target_idx = -1 - look_ahead 
        corrected_frame_6d = output_tensor[0, target_idx, :].cpu().numpy()
        
        return corrected_frame_6d

    # (가상의 데이터 스트림 시뮬레이션)
    # while True:
    #     raw_data = receive_osc()
    #     new_frame = convert_to_6d(raw_data)
    #     corrected_frame = process_incoming_vmc_frame(new_frame)
    #     final_quat = convert_to_quaternion(corrected_frame)
    #     send_osc_to_warudo(final_quat)

if __name__ == "__main__":
    main_server()
'''
