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
            
        if line_clean.startswith("OFFSET") and not is_end_site:
            parts = line_clean.split()
            offset_vec = [float(parts[1]), float(parts[2]), float(parts[3])]
            offsets.append(offset_vec)
            
    offsets_np = np.array(offsets, dtype=np.float32)
    offsets_np[:, 0] = -offsets_np[:, 0]
    return offsets_np

def parse_bvh_motion(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    motion_idx = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), -1)
    if motion_idx == -1:
        raise ValueError("MOTION 섹션을 찾을 수 없습니다.")
        
    data_lines = lines[motion_idx + 3:]
    motion_data = [[float(x) for x in line.strip().split()] for line in data_lines if line.strip()]
    return np.array(motion_data, dtype=np.float32)

def process_motion_for_unity(raw_motion):
    num_frames = raw_motion.shape[0]
    num_joints = raw_motion.shape[1] // 6
    reshaped_data = raw_motion.reshape(num_frames, num_joints, 6)
    
    positions = reshaped_data[:, :, 0:3]
    flat_euler = reshaped_data[:, :, 3:6].reshape(-1, 3)
    
    quaternions = R.from_euler('zxy', flat_euler, degrees=True).as_quat()
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    
    unity_positions = positions.copy()
    unity_quaternions = quaternions.copy()
    
    unity_positions[:, :, 0] = -unity_positions[:, :, 0]
    unity_quaternions[:, :, 1] = -unity_quaternions[:, :, 1]
    unity_quaternions[:, :, 2] = -unity_quaternions[:, :, 2]
    
    features = np.concatenate([unity_positions, unity_quaternions], axis=2)
    return features.reshape(num_frames, -1)

# 💡 수정된 부분: seq_length와 stride가 사라졌습니다.
def create_full_motion_dataset(bvh_dir, save_dir):
    """
    슬라이딩 윈도우 없이 1개의 BVH를 1개의 Unity 변환된 NPY 파일로 통째로 저장합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
    bvh_files.sort()
    print(f"총 {len(bvh_files)}개의 BVH 파일을 전체 모션 .npy로 변환합니다...")
    
    # 뼈대 정보(OFFSET)는 여전히 파이프라인에 필요하므로 추출해 둡니다.
    print("가장 첫 번째 파일에서 아바타의 골격(OFFSET) 정보를 추출합니다...")
    first_file_path = os.path.join(bvh_dir, bvh_files[0])
    skeleton_offsets = parse_bvh_offsets(first_file_path)
    
    offset_save_path = os.path.join(save_dir, "skeleton_offsets.npy")
    np.save(offset_save_path, skeleton_offsets)
    print(f"✅ 골격 정보 저장 완료! shape: {skeleton_offsets.shape}")
    
    for file in tqdm(bvh_files, desc="Converting Full BVH"):
        filepath = os.path.join(bvh_dir, file)
        save_path = os.path.join(save_dir, file.replace('.bvh', '.npy'))
        
        try:
            raw_motion = parse_bvh_motion(filepath)
            
            # features의 shape는 (전체 프레임 수, 관절수 * 7)이 됩니다.
            features = process_motion_for_unity(raw_motion)
            
            # 💡 조각내지 않고 features 배열 자체를 그대로 저장합니다.
            np.save(save_path, features)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"✅ 전체 모션 변환 완료! 모든 .npy 파일이 '{save_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    INPUT_DIR = r"Warudo_Send_Temp" 
    
    # 윈도우된 데이터와 폴더가 섞이지 않도록 이름을 명확히 다르게 지정하시는 것을 추천합니다.
    OUTPUT_DIR = r"Warudo_Send_Temp/Sample_Data" 
    
    create_full_motion_dataset(INPUT_DIR, OUTPUT_DIR)