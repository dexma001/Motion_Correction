import bvhio
import numpy as np

# --- 설정 ---
BVH_FILE_PATH = "bandai_dataset_motion.bvh"
NPY_FILE_PATH = "unity_ready_motion.npy"

# 반다이 남코 BVH 파일에 맞춘 BONE_MAPPING
BONE_MAPPING = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Chest": "Chest",
    # BVH에 UpperChest(Spine2)가 없으므로 생략합니다. Warudo(VRM)는 Chest만 있어도 작동합니다.
    "Neck": "Neck",
    "Head": "Head",
    "Shoulder_L": "LeftShoulder",
    "UpperArm_L": "LeftUpperArm",
    "LowerArm_L": "LeftLowerArm",
    "Hand_L": "LeftHand",
    "Shoulder_R": "RightShoulder",
    "UpperArm_R": "RightUpperArm",
    "LowerArm_R": "RightLowerArm",
    "Hand_R": "RightHand",
    "UpperLeg_L": "LeftUpperLeg",
    "LowerLeg_L": "LeftLowerLeg",
    "Foot_L": "LeftFoot",
    "Toes_L": "LeftToes",
    "UpperLeg_R": "RightUpperLeg",
    "LowerLeg_R": "RightLowerLeg",
    "Foot_R": "RightFoot",
    "Toes_R": "RightToes"
}

def convert_to_unity_coords(pos, quat):
    """BVH (Z-up/Right-Handed) -> Unity (Y-up/Left-Handed) 변환"""
    px, py, pz = -pos[0], pos[1], pos[2]
    qx, qy, qz, qw = quat.X, -quat.Y, -quat.Z, quat.W
    return (px, py, pz), (qx, qy, qz, qw)

def main():
    print("Loading BVH...")
    bvh_root = bvhio.readAsBvh(BVH_FILE_PATH)
    
    # 1. 스케일 보정 (cm -> m)
    bvh_root.Root.Scale = (0.01, 0.01, 0.01)
    
    frames = bvh_root.Frames
    frame_time = bvh_root.FrameTime
    
    unity_bone_names = list(BONE_MAPPING.values())
    num_bones = len(unity_bone_names)
    
    # AI 학습과 송신에 최적화된 Numpy 배열 생성
    # 형태: (전체 프레임 수, 뼈대 개수, 데이터 차원)
    positions = np.zeros((frames, num_bones, 3), dtype=np.float32)
    rotations = np.zeros((frames, num_bones, 4), dtype=np.float32)
    
    # 뼈대 참조를 캐싱하여 검색 속도 향상
    joint_refs = {}
    for joint in bvh_root.Root.layout():
        if joint.Name in BONE_MAPPING:
            joint_refs[BONE_MAPPING[joint.Name]] = joint

    print(f"Processing {frames} frames...")
    for frame_idx in range(frames):
        bvh_root.applyFrame(frame_idx)
        
        for bone_idx, unity_name in enumerate(unity_bone_names):
            # BVH에 해당 뼈대가 존재하면 변환 적용
            joint = joint_refs.get(unity_name)
            if joint:
                p, q = convert_to_unity_coords(joint.Position, joint.Rotation)
                positions[frame_idx, bone_idx] = p
                rotations[frame_idx, bone_idx] = q

    # 딕셔너리 형태로 묶어서 하나의 npy 파일로 저장
    dataset = {
        'bone_names': unity_bone_names,
        'positions': positions,
        'rotations': rotations,
        'frame_time': frame_time
    }
    
    np.save(NPY_FILE_PATH, dataset)
    print(f"Saved optimized dataset to {NPY_FILE_PATH}")
    print(f"Position Array Shape: {positions.shape}")
    print(f"Rotation Array Shape: {rotations.shape}")

if __name__ == "__main__":
    main()