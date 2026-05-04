import numpy as np

data_dict = np.load("temp.npz")

print("Keys in npz:", data_dict.files)

# 3. 본 이름 리스트 확인
bone_names = data_dict['names']
print("\n--- Bone Names ---")
print(bone_names)

# 4. 모션 데이터 구조 확인
motion_data = data_dict['data']
print("\n--- Motion Data Shape ---")
print(f"Shape: {motion_data.shape}") # (프레임 수, 본 개수 * 7)

