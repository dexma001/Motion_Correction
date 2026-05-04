from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
import numpy as np
import time

IP = "127.0.0.1"
PORT = 39539
# 🚨 본인의 VMC_Processed 폴더 내 실제 파일 경로로 반드시 수정하세요!
TEST_FILE = "./VMC_Processed/01/01_01_stages_ii_vmc.npy" 

def send_to_warudo():
    client = udp_client.SimpleUDPClient(IP, PORT)
    
    print(f"[{TEST_FILE}] 데이터 로드 중...")
    try:
        frames = np.load(TEST_FILE, allow_pickle=True)
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print("✅ Warudo로 '번들(Bundle)' 전송을 시작합니다... (종료하려면 Ctrl+C)")
    
    try:
        for frame_idx, frame_data in enumerate(frames):
            if frame_idx % 30 == 0:
                print(f"전송 중... {frame_idx} / {len(frames)} 프레임")

            # 📦 1. 하나의 소포 박스(Bundle) 생성
            bundle = OscBundleBuilder(time.time())

            # 2. 생존 신고 메시지 포장
            msg_ok = OscMessageBuilder(address="/VMC/Ext/OK")
            msg_ok.add_arg(1)
            bundle.add_content(msg_ok.build())

            # 3. 시간 동기화 메시지 포장
            msg_t = OscMessageBuilder(address="/VMC/Ext/T")
            msg_t.add_arg(float(time.time()))
            bundle.add_content(msg_t.build())

            # 4. 뼈대 데이터들을 순서대로 포장
            for bone_name, transform in frame_data.items():
                px, py, pz = transform["pos"]
                qx, qy, qz, qw = transform["rot"]
                fx, fy, fz = float(px), float(py), float(pz)
                fqx, fqy, fqz, fqw = float(qx), float(qy), float(qz), float(qw)

                # 골반(Root) 전용 메시지
                if bone_name == "Hips":
                    msg_root = OscMessageBuilder(address="/VMC/Ext/Root/Pos")
                    msg_root.add_arg("root")
                    msg_root.add_arg(fx); msg_root.add_arg(fy); msg_root.add_arg(fz)
                    msg_root.add_arg(fqx); msg_root.add_arg(fqy); msg_root.add_arg(fqz); msg_root.add_arg(fqw)
                    bundle.add_content(msg_root.build())
                
                # 개별 뼈대 메시지
                msg_bone = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
                msg_bone.add_arg(bone_name)
                msg_bone.add_arg(fx); msg_bone.add_arg(fy); msg_bone.add_arg(fz)
                msg_bone.add_arg(fqx); msg_bone.add_arg(fqy); msg_bone.add_arg(fqz); msg_bone.add_arg(fqw)
                bundle.add_content(msg_bone.build())
            
            # 🚚 5. 완성된 소포 박스를 한 번에 Warudo로 발송!
            client.send(bundle.build())
            
            time.sleep(0.033) # 약 30 FPS 속도
            
    except KeyboardInterrupt:
        print("\n🛑 전송을 중단했습니다.")

if __name__ == "__main__":
    send_to_warudo()