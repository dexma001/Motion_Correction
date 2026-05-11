import time
import numpy as np
from pythonosc import udp_client

# ==========================================================
# 💡 핵심 추가: Scipy(오른손) -> Unity(왼손) 좌표계 및 쿼터니언 변환기
# ==========================================================
def convert_to_unity_space(pos, quat):
    """
    오른손 좌표계(BVH/Scipy)를 왼손 좌표계(Unity)로 변환합니다.
    (일반적인 X축 반전 기준)
    """
    # 1. 위치(Position) 변환: X축을 반전시킵니다.
    unity_pos = [-pos[0], pos[1], pos[2]]
    
    # 2. 회전(Quaternion) 변환: 
    # 왼손 좌표계로 맞추기 위해 Y와 Z 성분을 반전시킵니다.
    # (엔진이나 익스포터 설정에 따라 -qx, qy, qz, -qw 가 될 수도 있습니다)
    unity_quat = [quat[0], -quat[1], -quat[2], quat[3]]
    
    return unity_pos, unity_quat

def main():
    WARUDO_IP = "127.0.0.1"
    WARUDO_PORT = 39539 
    DATA_PATH = r"Dataset/Bandai_Dataset_npys/dataset-1_walk-left_masculinity_001.npy" 
    
    client = udp_client.SimpleUDPClient(WARUDO_IP, WARUDO_PORT)
    print(f"Warudo VMC 서버({WARUDO_IP}:{WARUDO_PORT})에 연결되었습니다.")

    dataset = np.load(DATA_PATH)
    if dataset.ndim == 3:
        clip = np.vstack([dataset[:, 0, :], dataset[-1, 1:, :]])
    else:
        clip = dataset
        
    total_frames = clip.shape[0]
    num_joints = clip.shape[1] // 7
    
    # VMC Bone Names
    bone_names = [
        "Hips", "Spine", "Chest", "Neck", "Head", 
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
        "DummySite" 
    ]

    print("▶️ 모션 전송을 시작합니다... (Ctrl+C로 종료)")
    
    try:
        frame_idx = 0
        while True:
            current_frame = clip[frame_idx % total_frames].copy()
            joints_data = current_frame.reshape(num_joints, 7)
            
            # --- 1. Root (Hips) 전송 ---
            hips = joints_data[0]
            raw_pos = hips[0:3]
            raw_quat = hips[3:7]
            
            # 💡 Unity 공간으로 변환
            u_pos, u_quat = convert_to_unity_space(raw_pos, raw_quat)
            
            client.send_message(
                "/VMC/Ext/Root/Pos", 
                ["Hips", float(u_pos[0]), float(u_pos[1]), float(u_pos[2]), 
                 float(u_quat[0]), float(u_quat[1]), float(u_quat[2]), float(u_quat[3])]
            )
            
            # --- 2. 자식 관절 전송 ---
            for i in range(1, num_joints):
                bone_name = bone_names[i]
                bone = joints_data[i]
                
                # 💡 자식 뼈도 회전값 변환
                _, u_quat_bone = convert_to_unity_space([0,0,0], bone[3:7])
                
                client.send_message(
                    "/VMC/Ext/Bone/Pos", 
                    [bone_name, 0.0, 0.0, 0.0, 
                     float(u_quat_bone[0]), float(u_quat_bone[1]), float(u_quat_bone[2]), float(u_quat_bone[3])]
                )
            
            time.sleep(1/60)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n전송을 종료합니다.")

if __name__ == "__main__":
    main()