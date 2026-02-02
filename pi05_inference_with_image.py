import pickle
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pi05_infer import Pi05Inference

# ========== 图像预处理工具 ==========
def preprocess_image(image_path, size=224):
    """
    加载并预处理图像
    Args:
        image_path: 图像路径
        size: 目标尺寸
    Returns:
        归一化后的tensor (H, W, 3)
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # Resize到224x224
    img = img.resize((size, size), Image.BILINEAR)
    
    # 转为numpy数组
    img_array = np.array(img, dtype=np.float32)
    
    # 归一化到[0, 1]然后标准化
    # 使用ImageNet均值和标准差
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_array = (img_array - mean) / std
    
    # 转为torch tensor
    img_tensor = torch.from_numpy(img_array).to(torch.bfloat16)
    
    return img_tensor

# ========== 配置 ==========
number_of_images = 2  # 多视角观察
length_of_trajectory = 10  # 预测未来10步动作
checkpoint_path = "./converted_checkpoint_pi05.pkl"
tokenizer_path = "/home/x/Documents/models/paligemma-3b-pt-224/"

# ========== 加载模型 ==========
print("Loading checkpoint...")
converted_checkpoint = pickle.load(open(checkpoint_path, 'rb'))

print("Initializing model...")
infer = Pi05Inference(
    checkpoint=converted_checkpoint,
    num_views=number_of_images,
    chunk_size=length_of_trajectory,
    tokenizer_path=tokenizer_path,
    discrete_state_input=True,
)

# ========== 准备输入 ==========

# 方案1: 从图像文件加载（真实场景）
if False:  # 设为True使用真实图像
    image_paths = [
        "camera1_view.jpg",  # 第一个视角
        "camera2_view.jpg",  # 第二个视角
    ]
    
    images = []
    for img_path in image_paths:
        img_tensor = preprocess_image(img_path)
        images.append(img_tensor)
    
    normalized_observation_image_bfloat16 = torch.stack(images).cuda()

# 方案2: 使用dummy数据（测试）
else:
    print("Creating dummy observation images...")
    normalized_observation_image_bfloat16 = torch.randn(
        number_of_images, 224, 224, 3,
        dtype=torch.bfloat16,
        device='cuda'
    )

# 扩散噪声（标准正态分布）
print("Creating diffusion noise...")
diffusion_input_noise_bfloat16 = torch.randn(
    length_of_trajectory, 32,
    dtype=torch.bfloat16,
    device='cuda'
)

# 任务描述
task_prompt = "pick up the red cube and place it on the blue plate"

# 机器人状态（离散token表示）
# 示例：7-DOF机器人 + 夹爪状态
state_tokens = np.array([
    128,  # joint 1 angle (量化到0-255)
    95,   # joint 2 angle
    160,  # joint 3 angle
    110,  # joint 4 angle
    145,  # joint 5 angle
    200,  # joint 6 angle
    80,   # joint 7 angle
], dtype=np.int32)

# ========== 执行推理 ==========
print("\n" + "="*60)
print("Running inference...")
print("="*60)
print(f"Task: {task_prompt}")
print(f"Number of views: {number_of_images}")
print(f"Trajectory length: {length_of_trajectory}")
print(f"State tokens: {state_tokens}")
print()

with torch.no_grad():
    output_actions = infer.forward(
        observation_images_normalized=normalized_observation_image_bfloat16,
        diffusion_noise=diffusion_input_noise_bfloat16,
        task_prompt=task_prompt,
        state_tokens=state_tokens,
    )

print("\n" + "="*60)
print("Results")
print("="*60)
print(f"Output shape: {output_actions.shape}")  # (length_of_trajectory, 32)
print(f"Output dtype: {output_actions.dtype}")
print(f"\nPredicted actions (first 3 steps):")
print(output_actions)
print()

# 保存结果
torch.save(output_actions.cpu(), 'predicted_actions.pt')
print("Predicted actions saved to 'predicted_actions.pt'")
