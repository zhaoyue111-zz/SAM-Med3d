import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

# 设置后端
matplotlib.use('TkAgg')
# 指定文件夹路径
folder_path = './visul/'

# 获取所有 .nii.gz 文件
nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

# 计算总行数
total_images = len(nii_files)
rows = (total_images - 2) // 2 + 2  # 第一张和最后一张单独一行，其他每两张一行

# 创建一个 figure
fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))

# 显示第一张图片
ax = axes[0, 0]
file_path = os.path.join(folder_path, nii_files[0])
image = nib.load(file_path)
image_data = image.get_fdata()
slice_index = image_data.shape[2] // 2
ax.imshow(image_data[:, :, slice_index], cmap='gray')
ax.set_title(f'{nii_files[0]}')
ax.axis('off')

# 显示中间的图片，每两张一行
for i in range(1, total_images - 1):
    row = (i + 1) // 2
    col = (i + 1) % 2
    ax = axes[row, col]
    file_path = os.path.join(folder_path, nii_files[i])
    image = nib.load(file_path)
    image_data = image.get_fdata()
    slice_index = image_data.shape[2] // 2
    ax.imshow(image_data[:, :, slice_index], cmap='gray')
    ax.set_title(f'{nii_files[i]}')
    ax.axis('off')

# 显示最后一张图片
ax = axes[-1, 0]
file_path = os.path.join(folder_path, nii_files[-1])
image = nib.load(file_path)
image_data = image.get_fdata()
slice_index = image_data.shape[2] // 2
ax.imshow(image_data[:, :, slice_index], cmap='gray')
ax.set_title(f'{nii_files[-1]}')
ax.axis('off')

# 隐藏未使用的子图
for j in range(1, len(axes[-1])):
    axes[-1, j].axis('off')

# 调整布局
plt.tight_layout()
plt.show()