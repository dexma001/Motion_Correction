import numpy as np
import time
from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

IP = "127.0.0.1"
PORT = 39539
TEST_FILE = r"Dataset/Bandai_Dataset_Unity_Raw/dataset-1_bow_active_001.npy"

# ⚠️ 중요: npy 파일 생성 시 사용했던 22개 관절의 순서와 정확히 일치해야 합니다.
BONE_NAMES = [
        "Root",
        "Hips", "Spine", "Chest", "Neck", "Head", 
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
          
]

def send_to_warudo():
    client = udp_client.SimpleUDPClient(IP, PORT)
    SCALE = 0.01
    
    try:
        # 데이터 로드 (128, 154)
        frames = np.load(TEST_FILE)
        print(f"✅ 데이터 로드 완료: {frames.shape}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return

    print("🚀 Warudo로 전송 시작...")
    
    while True:
        try:
            for frame_idx, frame_data in enumerate(frames):
                bundle = OscBundleBuilder(time.time())

                # 생존 신고 및 시간 동기화
                msg_ok = OscMessageBuilder(address="/VMC/Ext/OK")
                msg_ok.add_arg(1)
                bundle.add_content(msg_ok.build())

                msg_t = OscMessageBuilder(address="/VMC/Ext/T")
                msg_t.add_arg(float(time.time()))
                bundle.add_content(msg_t.build())

                # 154개의 데이터를 7개씩 잘라서 처리
                for i, bone_name in enumerate(BONE_NAMES):
                    start_idx = i * 7
                    # pos(3) + rot(4)
                    px, py, pz = frame_data[start_idx : start_idx + 3] * SCALE
                    qx, qy, qz, qw = frame_data[start_idx + 3 : start_idx + 7]

                    # float 형변환 (numpy 타입을 일반 파이썬 float으로)
                    fx, fy, fz = map(float, [px, py, pz])
                    fqx, fqy, fqz, fqw = map(float, [qx, qy, qz, qw])

                    """
                    # Hips일 경우 Root Pos 추가 전송
                    if bone_name == "Hips":
                        msg_root = OscMessageBuilder(address="/VMC/Ext/Root/Pos")
                        msg_root.add_arg("root")
                        msg_root.add_arg(fx); msg_root.add_arg(fy); msg_root.add_arg(fz)
                        msg_root.add_arg(fqx); msg_root.add_arg(fqy); msg_root.add_arg(fqz); msg_root.add_arg(fqw)
                        bundle.add_content(msg_root.build())
                    """

                    # 일반 뼈대 데이터 전송
                    if bone_name != "Root":
                        msg_bone = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
                        msg_bone.add_arg(bone_name)
                        '''
                        msg_bone.add_arg(fx); msg_bone.add_arg(fy); msg_bone.add_arg(fz)
                        '''
                        if bone_name == "Hips":
                            msg_bone.add_arg(fx); msg_bone.add_arg(fy); msg_bone.add_arg(fz)
                        else:
                            msg_bone.add_arg(0.0); msg_bone.add_arg(0.0); msg_bone.add_arg(0.0)
                        
                        msg_bone.add_arg(fqx); msg_bone.add_arg(fqy); msg_bone.add_arg(fqz); msg_bone.add_arg(fqw)
                        bundle.add_content(msg_bone.build())

                client.send(bundle.build())
                
                if frame_idx % 30 == 0:
                    print(f"전송 중... {frame_idx}/{len(frames)}")
                
                time.sleep(1/15)

        except KeyboardInterrupt:
            print("\n🛑 중단되었습니다.")
            break

if __name__ == "__main__":
    send_to_warudo()