import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def parse_bvh_offsets(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    offsets = []
    is_end_site = False
    
    for line in lines:
        line_clean = line.strip()
        if line_clean == "MOTION":
            break
        
        if line_clean.startswith("End Site"):
            is_end_site = True
            continue
            
        if is_end_site and line_clean == "}":
            is_end_site = False
            continue    
            
        # OFFSET 키워드를 찾아서 X, Y, Z 좌표 추출
        if line_clean.startswith("OFFSET") and not is_end_site:
            parts = line_clean.split()
            # str -> float 변환
            offset_vec = [float(parts[1]), float(parts[2]), float(parts[3])]
            offsets.append(offset_vec)
            
    offsets_np = np.array(offsets, dtype=np.float32)
    
    # --- 🚨 중요: OFFSET도 Unity(왼손 좌표계)에 맞춰 X축 부호 반전 ---
    offsets_np[:, 0] = -offsets_np[:, 0]
    
    return offsets_np

def parse_bvh_motion(filepath):
    """BVH 파일에서 MOTION 데이터 파싱"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    motion_idx = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), -1)
    if motion_idx == -1:
        raise ValueError("MOTION 섹션을 찾을 수 없습니다.")
        
    data_lines = lines[motion_idx + 3:]
    motion_data = [[float(x) for x in line.strip().split()] for line in data_lines if line.strip()]
    
    return np.array(motion_data, dtype=np.float32)

def process_motion_for_unity(raw_motion):
    """
    [1단계 완수] 
    BVH 파싱 (오른손) -> Quaternion 변환 -> Unity 좌표계(왼손) 강제 변환
    """
    num_frames = raw_motion.shape[0]
    num_joints = raw_motion.shape[1] // 6
    reshaped_data = raw_motion.reshape(num_frames, num_joints, 6)
    
    positions = reshaped_data[:, :, 0:3]
    flat_euler = reshaped_data[:, :, 3:6].reshape(-1, 3)
    
    # 오른손 좌표계 기준 쿼터니언 변환 (BVH는 ZXY 오일러 사용)
    quaternions = R.from_euler('ZXY', flat_euler, degrees=True).as_quat()
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    
    # --- Unity (왼손 좌표계) 변환 ---
    unity_positions = positions.copy()
    unity_quaternions = quaternions.copy()
    
    # 1. 위치: X축 반전
    unity_positions[:, :, 0] = -unity_positions[:, :, 0]
    
    # 2. 회전: 쿼터니언 Y, Z 반전 (Unity는 x, y, z, w 구조에서 y, z를 뒤집음)
    unity_quaternions[:, :, 1] = -unity_quaternions[:, :, 1]
    unity_quaternions[:, :, 2] = -unity_quaternions[:, :, 2]
    
    '''
    bvh: Y_up Right Handed
    Unity: Y_up Left Handed
    '''
    
    # Feature 결합 (프레임, 관절수, 7차원)
    features = np.concatenate([unity_positions, unity_quaternions], axis=2)
    return features.reshape(num_frames, -1)

def create_windowed_dataset(bvh_dir, save_dir, seq_length=60, stride=3):
    """
    [3단계 완수] Stride가 적용된 시퀀스 윈도우 생성
    """

    bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
    bvh_files.sort()
    print(f"총 {len(bvh_files)}개의 BVH 파일을 개별 .npy로 변환합니다...")
    
    print("가장 첫 번째 파일에서 아바타의 골격(OFFSET) 정보를 추출합니다...")
    first_file_path = os.path.join(bvh_dir, bvh_files[0])
    skeleton_offsets = parse_bvh_offsets(first_file_path)
    
    offset_save_path = "skeleton_offsets.npy"
    np.save(offset_save_path, skeleton_offsets)
    print(f"✅ 골격 정보 저장 완료! shape: {skeleton_offsets.shape} -> '{offset_save_path}'")
    
    file_windows = []
    
    for file in tqdm(bvh_files, desc="Converting & Windowing BVH"):
        filepath = os.path.join(bvh_dir, file)
        
        try:
            raw_motion = parse_bvh_motion(filepath)
            features = process_motion_for_unity(raw_motion)
            
            # --- Stride 3 적용 구간 ---
            # range(start, stop, step)을 사용하여 stride 구현
            for i in range(0, len(features) - seq_length + 1, stride):
                window = features[i : i + seq_length]
                file_windows.append(window)
                
        except Exception as e:
            print(f"Error processing {file}: {e}") 
                
    dataset = np.array(file_windows, dtype=np.float32)
    np.save(save_dir, dataset)
            
    print(f"✅ 개별 변환 완료! 모든 .npy 파일이 '{save_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    INPUT_DIR = r"Dataset/Bandai_Dataset" 
    OUTPUT_DIR = "Bandai_Dataset_Unity_advanced.npy" 
    
    # seq_length: 60프레임 (약 1~2초치 모션), stride: 3프레임 간격 건너뛰기
    create_windowed_dataset(INPUT_DIR, OUTPUT_DIR, seq_length=60, stride=3)