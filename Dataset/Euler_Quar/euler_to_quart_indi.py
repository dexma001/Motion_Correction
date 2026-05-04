import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm # 진행률 바 표시용

# ==============================================================================
# 원본과 동일하게 유지된 핵심 함수 2개 (절대 수정하지 않음)
# ==============================================================================
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
    """
    1. BVH 파싱 (오른손 좌표계) -> 2. Unity 좌표계(왼손)로 변환 -> 3. 결합
    """
    num_frames = raw_motion.shape[0]
    num_joints = raw_motion.shape[1] // 6
    reshaped_data = raw_motion.reshape(num_frames, num_joints, 6)
    
    # 1. Local Position & Local Euler
    positions = reshaped_data[:, :, 0:3]
    flat_euler = reshaped_data[:, :, 3:6].reshape(-1, 3)
    
    # 2. Scipy를 이용해 먼저 완벽한 쿼터니언(오른손 좌표계)으로 변환
    quaternions = R.from_euler('ZXY', flat_euler, degrees=True).as_quat()
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    
    # ==========================================================
    # 💡 3. Unity(왼손 좌표계) 강제 변환 (Baking)
    # ==========================================================
    unity_positions = positions.copy()
    unity_quaternions = quaternions.copy()
    
    # 위치 변환: X축 반전
    unity_positions[:, :, 0] = -unity_positions[:, :, 0]
    
    # 회전 변환: 쿼터니언의 Y, Z 성분 반전 [x, -y, -z, w]
    unity_quaternions[:, :, 1] = -unity_quaternions[:, :, 1]
    unity_quaternions[:, :, 2] = -unity_quaternions[:, :, 2]
    # ==========================================================
    
    # 4. Feature 결합: [Unity_PosX, Y, Z, Unity_QuatX, Y, Z, W]
    features = np.concatenate([unity_positions, unity_quaternions], axis=2)
    flat_features = features.reshape(num_frames, -1)
    
    return flat_features
'''

# ==============================================================================
# 수정된 부분: 단일 배열이 아닌 개별 파일로 저장하는 로직
# ==============================================================================
def create_individual_unity_datasets(bvh_dir, save_dir, seq_length=60):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
    print(f"총 {len(bvh_files)}개의 BVH 파일을 개별 .npy로 변환합니다...")
    
    # tqdm으로 진행 상태를 보여줌
    for file in tqdm(bvh_files, desc="Converting BVH"):
        filepath = os.path.join(bvh_dir, file)
        
        # 저장할 파일명 설정 (예: motion_01.bvh -> motion_01.npy)
        save_name = file.replace('.bvh', '.npy')
        save_path = os.path.join(save_dir, save_name)
        
        try:
            raw_motion = parse_bvh_motion(filepath)
            features = process_motion_for_unity(raw_motion)
            
            # 이 파일 1개에 대한 Sliding Window 데이터를 담을 리스트
            file_windows = []
            
            # Sliding Window
            for i in range(len(features) - seq_length):
                window = features[i : i + seq_length]
                file_windows.append(window)
                
            # 모션이 너무 짧아서 60프레임이 안 나오는 파일은 스킵
            if len(file_windows) == 0:
                continue
                
            # 1개 파일의 전체 윈도우를 Numpy 배열로 변환
            dataset = np.array(file_windows, dtype=np.float32)
            
            # 해당 파일을 개별 저장
            np.save(save_path, dataset)
            
        except Exception as e:
            print(f"\nError processing {file}: {e}")
            
    print("-" * 50)
    print(f"✅ 개별 변환 완료! 모든 .npy 파일이 '{save_dir}'에 저장되었습니다.")
    print("-" * 50)

if __name__ == "__main__":
    # 원본 BVH가 있는 폴더
    INPUT_DIR = r"Dataset/Bandai_Dataset" 
    
    # 개별 npy 파일들이 저장될 새로운 폴더
    OUTPUT_DIR = r"Dataset/Bandai_Dataset_npys" 
    
    if not os.path.exists(INPUT_DIR):
        print(f"'{INPUT_DIR}' 폴더를 찾을 수 없습니다.")
    else:
        create_individual_unity_datasets(INPUT_DIR, OUTPUT_DIR, seq_length=60)