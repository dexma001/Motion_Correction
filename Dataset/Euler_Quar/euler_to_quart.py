import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_bvh_motion(filepath):
    """
    BVH 파일에서 MOTION 데이터만 파싱합니다.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    motion_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_idx = i
            break
            
    if motion_idx == -1:
        raise ValueError("MOTION 섹션을 찾을 수 없습니다.")
        
    data_lines = lines[motion_idx + 3:]
    motion_data = [[float(x) for x in line.strip().split()] for line in data_lines if line.strip()]
    
    return np.array(motion_data, dtype=np.float32)

#Original
def process_motion_for_unity(raw_motion):
    """
    Unity의 Transform(.localPosition, .localRotation)에 1:1 대응하도록
    관절당 7개의 Feature(Pos 3차원 + Quat 4차원)로 변환합니다.
    """
    num_frames = raw_motion.shape[0]
    
    # 관절 당 6개의 채널(Pos 3 + Euler 3)이 있다고 가정
    num_joints = raw_motion.shape[1] // 6
    reshaped_data = raw_motion.reshape(num_frames, num_joints, 6)
    
    # 1. Local Position 유지 (X, Y, Z)
    positions = reshaped_data[:, :, 0:3]
    
    # 2. Local Euler Angles 추출 (Z, X, Y)
    rotations_euler = reshaped_data[:, :, 3:6]
    
    # 3. Euler -> Quaternion 변환 (Unity 규격: x, y, z, w)
    flat_euler = rotations_euler.reshape(-1, 3)
    # BVH의 ZXY 순서를 명시하여 쿼터니언으로 변환
    quaternions = R.from_euler('ZXY', flat_euler, degrees=True).as_quat()
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    
    # 4. Feature 결합: [PosX, PosY, PosZ, QuatX, QuatY, QuatZ, QuatW]
    # 관절당 7차원 데이터 생성
    features = np.concatenate([positions, quaternions], axis=2)
    
    # (프레임 수, 관절 수 * 7) 로 평탄화
    flat_features = features.reshape(num_frames, -1)
    
    return flat_features

'''
#_2
# euler_to_quart.py 에 덮어씌울 함수 (indi와 동일하게 맞춤)
def process_motion_for_unity(raw_motion):
    num_frames = raw_motion.shape[0]
    num_joints = raw_motion.shape[1] // 6
    reshaped_data = raw_motion.reshape(num_frames, num_joints, 6)
    
    positions = reshaped_data[:, :, 0:3]
    flat_euler = reshaped_data[:, :, 3:6].reshape(-1, 3)
    
    quaternions = R.from_euler('ZXY', flat_euler, degrees=True).as_quat()
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    
    # --- 💡 Unity(왼손 좌표계) 강제 변환 (Baking) ---
    unity_positions = positions.copy()
    unity_quaternions = quaternions.copy()
    
    # 위치 변환: X축 반전
    unity_positions[:, :, 0] = -unity_positions[:, :, 0]
    # 회전 변환: 쿼터니언의 Y, Z 성분 반전
    unity_quaternions[:, :, 1] = -unity_quaternions[:, :, 1]
    unity_quaternions[:, :, 2] = -unity_quaternions[:, :, 2]
    # -----------------------------------------------
    
    features = np.concatenate([unity_positions, unity_quaternions], axis=2)
    flat_features = features.reshape(num_frames, -1)
    
    return flat_features
'''

def create_unity_dataset(bvh_dir, save_path, seq_length=60):
    all_windows = []
    bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
    print(f"총 {len(bvh_files)}개의 BVH 파일을 Unity 전용으로 변환합니다...")
    
    for file in bvh_files:
        filepath = os.path.join(bvh_dir, file)
        
        try:
            raw_motion = parse_bvh_motion(filepath)
            features = process_motion_for_unity(raw_motion)
            
            # Sliding Window
            for i in range(len(features) - seq_length):
                window = features[i : i + seq_length]
                all_windows.append(window)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    # 전체 데이터를 Numpy 배열로 변환
    dataset = np.array(all_windows, dtype=np.float32)
    
    # ❗주의: Unity 적용을 위해 데이터 정규화(Normalization)를 의도적으로 생략합니다.
    # 쿼터니언의 단위 벡터 성질과 모델의 물리적 공간감을 보존하기 위함입니다.
    
    np.save(save_path, dataset)
    
    print("-" * 50)
    print(f"✅ Unity용 변환 완료! 최종 데이터 Shape: {dataset.shape}")
    print(f"✅ 저장된 파일: {save_path}")
    print(f"💡 모델의 In_Channels는 {dataset.shape[2]} 로 설정해야 합니다.")
    print("-" * 50)

if __name__ == "__main__":
    INPUT_DIR = r"Dataset/Bandai_Dataset" 
    OUTPUT_FILE = "Bandai_Dataset_Unity.npy"
    
    if not os.path.exists(INPUT_DIR):
        print(f"'{INPUT_DIR}' 폴더를 찾을 수 없습니다.")
    else:
        create_unity_dataset(INPUT_DIR, OUTPUT_FILE, seq_length=60)