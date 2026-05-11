import time
import numpy as np
from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

WARUDO_IP = "127.0.0.1"
WARUDO_PORT = 39539
NPY_FILE_PATH = "unity_ready_motion.npy"

def main():
    # 1. 전처리된 NPY 데이터셋 로드 (딕셔너리 형태이므로 allow_pickle=True 필요)
    print(f"Loading {NPY_FILE_PATH}...")
    dataset = np.load(NPY_FILE_PATH, allow_pickle=True).item()
    
    bone_names = dataset['bone_names']
    positions = dataset['positions']
    rotations = dataset['rotations']
    frame_time = dataset['frame_time']
    frames = positions.shape[0]
    
    # OSC 클라이언트 준비
    client = udp_client.SimpleUDPClient(WARUDO_IP, WARUDO_PORT)
    print(f"Started Streaming... Total Frames: {frames}")
    
    start_time = time.time()

    try:
        for frame_idx in range(frames):
            loop_start = time.time()
            
            bundle = OscBundleBuilder(OscBundleBuilder.IMMEDIATELY)

            # 시간 정보
            msg_time = OscMessageBuilder(address="/VMC/Ext/T")
            msg_time.add_arg(float(time.time() - start_time))
            bundle.add_content(msg_time.build())

            # Root (기본값 전송)
            msg_root = OscMessageBuilder(address="/VMC/Ext/Root/Pos")
            msg_root.add_arg("root")
            for val in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]:
                msg_root.add_arg(val)
            bundle.add_content(msg_root.build())

            # ★ 핵심: 이미 Unity 규격으로 정제된 배열(Numpy)에서 값만 빼서 즉시 송신 ★
            for bone_idx, bone_name in enumerate(bone_names):
                px, py, pz = positions[frame_idx, bone_idx]
                qx, qy, qz, qw = rotations[frame_idx, bone_idx]
                
                # VMC에서 에러가 나지 않도록 float으로 변환하여 전송 (numpy.float32 방지)
                msg_bone = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
                msg_bone.add_arg(bone_name)
                msg_bone.add_arg(float(px))
                msg_bone.add_arg(float(py))
                msg_bone.add_arg(float(pz))
                msg_bone.add_arg(float(qx))
                msg_bone.add_arg(float(qy))
                msg_bone.add_arg(float(qz))
                msg_bone.add_arg(float(qw))
                bundle.add_content(msg_bone.build())

            client.send(bundle.build())

            # 프레임 타임에 맞춰 대기
            processing_time = time.time() - loop_start
            sleep_time = frame_time - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStreaming stopped.")

if __name__ == "__main__":
    main()