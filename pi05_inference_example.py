import pickle
import torch
import numpy as np
from pi05_infer import Pi05Inference

# ========== 配置参数 ==========
number_of_images = 2  # 观察视角数量
length_of_trajectory = 10  # 轨迹长度（动作序列长度）
image_size = 224  # 图像尺寸
action_dim = 32  # 动作维度

# ========== 加载checkpoint ==========
print("Loading checkpoint...")
converted_checkpoint = pickle.load(open('./converted_checkpoint_pi05.pkl', 'rb'))

# ========== 初始化模型 ==========
print("Initializing Pi05Inference...")
infer = Pi05Inference(
    checkpoint=converted_checkpoint,
    num_views=number_of_images,
    chunk_size=length_of_trajectory,
    tokenizer_path="/home/x/Documents/models/paligemma-3b-pt-224/",  # 需要下载paligemma tokenizer
    discrete_state_input=True,  # 推荐使用离散状态输入
    max_tokenize_len=200,
)

# ========== 构造输入数据 ==========

# 1. 观察图像（已归一化）
# Shape: (number_of_images, 224, 224, 3)
# 假设图像已经过预处理和归一化
print("Creating dummy observation images...")
normalized_observation_image_bfloat16 = torch.randn(
    number_of_images, image_size, image_size, 3,
    dtype=torch.bfloat16,
    device='cuda'
)

# 2. 扩散噪声输入
# Shape: (length_of_trajectory, 32)
# 用于diffusion-based动作生成
print("Creating diffusion noise...")
diffusion_input_noise_bfloat16 = torch.randn(
    length_of_trajectory, action_dim,
    dtype=torch.bfloat16,
    device='cuda'
)

# 3. 任务提示文本
task_prompt = "pick up the red cube and place it on the blue plate"

# 4. 状态tokens（离散表示）
# 这些是机器人状态的离散化表示（如关节角度、夹爪状态等）
# 假设状态空间已经量化为0-255的整数
print("Creating state tokens...")
state_dim = 7  # 例如：7自由度机器人
state_tokens = np.random.randint(0, 256, size=state_dim, dtype=np.int32)

# ========== 执行推理 ==========
print("\nRunning inference...")
print(f"Task: {task_prompt}")
print(f"State tokens: {state_tokens}")
print(f"Input shapes:")
print(f"  - Images: {normalized_observation_image_bfloat16.shape}")
print(f"  - Noise: {diffusion_input_noise_bfloat16.shape}")

output_actions = infer.forward(
    observation_images_normalized=normalized_observation_image_bfloat16,
    diffusion_noise=diffusion_input_noise_bfloat16,
    task_prompt=task_prompt,
    state_tokens=state_tokens,
)

print(f"\nOutput actions shape: {output_actions.shape}")
print(f"Output actions dtype: {output_actions.dtype}")
print(f"Output actions (first 5 steps):\n{output_actions}")
