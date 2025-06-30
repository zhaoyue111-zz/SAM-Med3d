"""
Modified version of infer_sequence.py that supports user-provided points for inference.
Maintains all original functionality while adding support for custom point input.

主要功能：
1. 支持自动生成点进行分割（auto模式）
2. 支持用户自定义点进行分割（process_3D_image模式）
3. 使用滑动窗口处理大尺寸图像
4. 保存分割结果和点信息
"""
import json
import os
import os.path as osp
import argparse
import pickle
import sys
from collections import OrderedDict, defaultdict
from glob import glob
from itertools import product
from os.path import join
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from utils.click_method import get_next_click3D_torch_no_gt, get_next_click3D_torch_no_gt_naive
from utils.data_loader import Dataset_Union_ALL_Infer

# Constants
CLICK_METHODS = {
    "no_gt": get_next_click3D_torch_no_gt,
    "no_gt_naive": get_next_click3D_torch_no_gt_naive,
}


def init_args(params):
    """Initialize arguments with default values."""

    class Args:
        test_data_path = params.get("test_data_path", "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/data/validation")
        checkpoint_path = params.get("checkpoint_path", "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/ckpt/sam_med3d_turbo.pth")
        output_dir = params.get("output_dir", "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/visualization/test_amos")
        pred_output_dir = join(output_dir, "pred")
        save_image = params.get("save_image", True)
        sliding_window = params.get("sliding_window", False)
        image_size = params.get("image_size", 256)
        crop_size = params.get("crop_size", 128)
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model_type = params.get("model_type", "vit_b_ori")
        num_clicks = params.get("num_clicks", 5)
        point_method = params.get("point_method", "no_gt")
        threshold = params.get("threshold", 0)
        dim = params.get("dim", 3)
        seed = params.get("seed", 2023)
        dot_list = params.get("dot_list", None)
        skip_existing_pred = params.get("skip_existing_pred", False)

    args = Args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args


def load_model(args):
    """Load and initialize the model."""
    model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    if os.path.exists(args.checkpoint_path):
        model_dict = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(model_dict["model_state_dict"], strict=False)
    return model


def get_save_paths(meta_info: Dict[str, Any], args: Any) -> Tuple[str, str, str]:
    """获取保存路径

    Args:
        meta_info: 图像元信息
        args: 参数配置

    Returns:
        vis_root: 可视化结果根目录
        pred_path: 预测结果保存路径
        img_name: 图像名称
    """
    img_name = meta_info["image_path"][0]
    modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
    dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
    vis_root = osp.join(args.pred_output_dir, modality, dataset)
    pred_path = osp.join(
        vis_root,
        osp.basename(img_name).replace(".nii.gz", f"_pred{args.num_clicks - 1}.nii.gz")
    )
    return vis_root, pred_path, img_name


def init_result_dict(pred_path: str) -> Dict[str, Any]:
    """初始化结果字典

    Args:
        pred_path: 预测结果保存路径

    Returns:
        result: 包含masks、points、labels和pred_path的字典
    """
    return {
        "masks": [],      # 存储所有预测的掩码
        "points": None,   # 存储点击点坐标 [array([[[x, y, z]]])]
        "labels": None,   # 存储点击点标签 [array([[label]])]
        "pred_path": pred_path
    }


def setup_sliding_window(image3D: torch.Tensor, args: Any) -> Tuple[Dict[int, np.ndarray], List[Tuple[torch.Tensor, Dict[str, Any]]]]:
    """设置滑动窗口处理

    Args:
        image3D: 输入的3D图像
        args: 参数配置

    Returns:
        pred3D_full_dict: 预测结果字典
        sliding_window_list: 滑动窗口列表
    """
    pred3D_full_dict = {
        click_idx: torch.zeros_like(image3D).numpy()
        for click_idx in range(args.num_clicks)
    }

    offset_mode = "center" if (not args.sliding_window) else "rounded"
    crop_transform = tio.CropOrPad(
        target_shape=(args.crop_size, args.crop_size, args.crop_size)
    )
    sliding_window_list = pad_and_crop_with_sliding_window(
        image3D, crop_transform, offset_mode=offset_mode
    )

    return pred3D_full_dict, sliding_window_list


def save_results(result: Dict[str, Any], image3D: torch.Tensor, meta_info: Dict[str, Any],
                vis_root: str, img_name: str, click_points: List[np.ndarray],
                click_labels: List[np.ndarray], pred3D_full_dict: Dict[int, np.ndarray],
                args: Any) -> None:
    """保存处理结果

    Args:
        result: 结果字典
        image3D: 输入的3D图像
        meta_info: 图像元信息
        vis_root: 可视化结果根目录
        img_name: 图像名称
        click_points: 点击点列表
        click_labels: 点击标签列表
        pred3D_full_dict: 预测结果字典
        args: 参数配置
    """
    os.makedirs(vis_root, exist_ok=True)

    # 保存点信息
    pt_info = dict(points=click_points, labels=click_labels)
    pt_path = osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_pt.pkl"))
    pickle.dump(pt_info, open(pt_path, "wb"))

    # 保存原始图像
    if args.save_image:
        save_numpy_to_nifti(
            image3D,
            osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_img.nii.gz")),
            meta_info,
        )

    # 保存预测结果
    for idx, pred3D_full in pred3D_full_dict.items():
        # 只保存非零部分
        non_zero_indices = np.nonzero(pred3D_full)
        if len(non_zero_indices[0]) > 0:
            mask_info = {
                "indices": [indices.tolist() for indices in non_zero_indices],
                "values": pred3D_full[non_zero_indices].tolist(),
                "shape": pred3D_full.shape
            }
            result["masks"].append(mask_info)
        else:
            result["masks"].append(None)

        # 保存不带点的预测掩码
        save_numpy_to_nifti(
            pred3D_full,
            osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{idx}.nii.gz")),
            meta_info,
        )

        # 在预测掩码上添加点并保存
        pred_with_points = pred3D_full.copy()
        radius = 2
        for pt in click_points[:idx + 1]:
            pred_with_points[
                ...,
                int(pt[0, 0, 0] - radius):int(pt[0, 0, 0] + radius),
                int(pt[0, 0, 1] - radius):int(pt[0, 0, 1] + radius),
                int(pt[0, 0, 2] - radius):int(pt[0, 0, 2] + radius),
            ] = 10

        save_numpy_to_nifti(
            pred_with_points,
            osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{idx}_wPt.nii.gz")),
            meta_info,
        )


def process_3D_image(image3D: torch.Tensor, model: torch.nn.Module, num_clicks: int,
                    meta_info: Dict[str, Any], args: Any, output_path: str, dot_list: List[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """处理3D图像，支持用户自定义点进行分割

    Args:
        image3D: 输入的3D图像
        model: SAM模型
        num_clicks: 点击次数
        meta_info: 图像元信息
        args: 参数配置
        output_path: JSON文件保存路径
        dot_list: 用户自定义的点列表，格式为[(x, y, z, label), ...]

    Returns:
        result: 包含masks、points、labels和pred_path的字典
    """
    # 获取保存路径
    vis_root, pred_path, img_name = get_save_paths(meta_info, args)

    # 初始化结果字典
    result = init_result_dict(pred_path)

    if args.skip_existing_pred and osp.exists(pred_path):
        return result

    # 设置滑动窗口
    pred3D_full_dict, sliding_window_list = setup_sliding_window(image3D, args)

    click_points = []
    click_labels = []

    # 处理每个窗口
    for image3D_window, pos3D in sliding_window_list:
        # 图像预处理
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        image3D_window = norm_transform(image3D_window.squeeze(dim=1))
        image3D_window = image3D_window.unsqueeze(dim=1).float()

        if dot_list:
            for i in range(min(num_clicks, len(dot_list))):
                x, y, z, label = dot_list[i]
                # 创建点和标签张量
                points_co = torch.tensor([[[x, y, z]]], dtype=torch.float).to(model.device)
                points_la = torch.tensor([[label]], dtype=torch.int64).to(model.device)

                # 获取图像特征
                with torch.no_grad():
                    image_embeddings = model.image_encoder(image3D_window.to(model.device))

                # 初始化掩码
                prev_masks = torch.zeros_like(image3D_window).to(model.device)
                low_res_masks = F.interpolate(
                    prev_masks.float(),
                    size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4),
                )

                # 获取嵌入
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=[points_co, points_la],
                    boxes=None,
                    masks=low_res_masks.to(model.device),
                )

                # 解码掩码
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings.to(model.device),
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # 后处理掩码
                prev_masks = F.interpolate(
                    low_res_masks,
                    size=image3D_window.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

                # 生成预测
                medsam_seg_prob = torch.sigmoid(prev_masks)
                medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

                # 处理ROI区域
                ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]
                seg_mask_roi = medsam_seg[
                    ...,
                    pred_roi[0]: pred_roi[1],
                    pred_roi[2]: pred_roi[3],
                    pred_roi[4]: pred_roi[5],
                ]
                pred3D_full_dict[i][
                    ...,
                    ori_roi[0]: ori_roi[1],
                    ori_roi[2]: ori_roi[3],
                    ori_roi[4]: ori_roi[5],
                ] = seg_mask_roi
                # 每添加一个mask就保存到JSON
                result["masks"].append(seg_mask_roi)
                current_result = convert_ndarray(result)
                with open(output_path, "r", encoding='utf-8') as f:
                    result_list = json.load(f)
                result_list.append(current_result)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(result_list, f, ensure_ascii=False, indent=2)

                if i >= len(click_points):
                    click_points.append(points_co.cpu().numpy())
                    click_labels.append(points_la.cpu().numpy())
                    # 每添加一个点就保存到JSON
                    result["points"] = click_points
                    result["labels"] = click_labels
                    current_result = convert_ndarray(result)
                    with open(output_path, "r", encoding='utf-8') as f:
                        result_list = json.load(f)
                    result_list.append(current_result)
                    with open(output_path, "w", encoding='utf-8') as f:
                        json.dump(result_list, f, ensure_ascii=False, indent=2)

    # 保存结果
    save_results(result, image3D, meta_info, vis_root, img_name,
                click_points, click_labels, pred3D_full_dict, args)

    print("Done")
    return result


def save_numpy_to_nifti(in_arr, out_path, meta_info):
    """Save numpy array as NIFTI file with metadata."""
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def pad_and_crop_with_sliding_window(img3D, crop_transform, offset_mode="center"):
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img3D.squeeze(0)),
    )
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)

    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)

    roi_shape = crop_transform.target_shape
    vol_bound = (0, img3D.shape[2], 0, img3D.shape[3], 0, img3D.shape[4])

    # 增加滑动窗口的重叠区域
    overlap = 32  # 重叠区域大小

    # 计算步长
    stride = roi_shape[0] - overlap

    # 计算在每个维度上需要的窗口数
    n_x = max(1, int(np.ceil((img3D.shape[2] - overlap) / stride)))
    n_y = max(1, int(np.ceil((img3D.shape[3] - overlap) / stride)))
    n_z = max(1, int(np.ceil((img3D.shape[4] - overlap) / stride)))

    window_list = []

    # 在三个维度上滑动窗口
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                # 计算当前窗口的起始位置
                x_start = min(i * stride, img3D.shape[2] - roi_shape[0])
                y_start = min(j * stride, img3D.shape[3] - roi_shape[1])
                z_start = min(k * stride, img3D.shape[4] - roi_shape[2])

                # 计算当前窗口的结束位置
                x_end = min(x_start + roi_shape[0], img3D.shape[2])
                y_end = min(y_start + roi_shape[1], img3D.shape[3])
                z_end = min(z_start + roi_shape[2], img3D.shape[4])

                # 创建当前窗口的padding参数
                padding_params = [0] * 6
                if x_start < 0:
                    padding_params[0] = -x_start
                if x_end > img3D.shape[2]:
                    padding_params[1] = x_end - img3D.shape[2]
                if y_start < 0:
                    padding_params[2] = -y_start
                if y_end > img3D.shape[3]:
                    padding_params[3] = y_end - img3D.shape[3]
                if z_start < 0:
                    padding_params[4] = -z_start
                if z_end > img3D.shape[4]:
                    padding_params[5] = z_end - img3D.shape[4]

                # 创建裁剪参数
                cropping_params = (
                    max(0, x_start),
                    img3D.shape[2] - min(img3D.shape[2], x_end),
                    max(0, y_start),
                    img3D.shape[3] - min(img3D.shape[3], y_end),
                    max(0, z_start),
                    img3D.shape[4] - min(img3D.shape[4], z_end),
                )

                # 应用padding和裁剪
                pad_and_crop = tio.Compose([
                    tio.Pad(padding_params, padding_mode=crop_transform.padding_mode),
                    tio.Crop(cropping_params),
                ])

                subject_roi = pad_and_crop(subject)
                img3D_roi = subject_roi.image.data.clone().detach().unsqueeze(1)

                # 记录位置信息
                pos3D_roi = {
                    'padding_params': padding_params,
                    'cropping_params': cropping_params,
                    'ori_roi': (x_start, x_end, y_start, y_end, z_start, z_end),
                    'pred_roi': (
                        padding_params[0],
                        roi_shape[0] - padding_params[1],
                        padding_params[2],
                        roi_shape[1] - padding_params[3],
                        padding_params[4],
                        roi_shape[2] - padding_params[5],
                    ),
                    'weight': np.ones(roi_shape),  # 添加权重信息用于后续融合
                }

                # 在重叠区域应用渐变权重
                if overlap > 0:
                    # 创建三维高斯权重
                    for axis in range(3):
                        pos = np.arange(roi_shape[axis])
                        if i > 0 and axis == 0:  # 左边重叠
                            w = 0.5 * (1 + np.cos(np.pi * (overlap - pos[:overlap]) / overlap))
                            pos3D_roi['weight'][:overlap, :, :] *= w[:, None, None]
                        if i < n_x - 1 and axis == 0:  # 右边重叠
                            w = 0.5 * (1 + np.cos(np.pi * pos[-overlap:] / overlap))
                            pos3D_roi['weight'][-overlap:, :, :] *= w[:, None, None]
                        if j > 0 and axis == 1:  # 前面重叠
                            w = 0.5 * (1 + np.cos(np.pi * (overlap - pos[:overlap]) / overlap))
                            pos3D_roi['weight'][:, :overlap, :] *= w[None, :, None]
                        if j < n_y - 1 and axis == 1:  # 后面重叠
                            w = 0.5 * (1 + np.cos(np.pi * pos[-overlap:] / overlap))
                            pos3D_roi['weight'][:, -overlap:, :] *= w[None, :, None]
                        if k > 0 and axis == 2:  # 上面重叠
                            w = 0.5 * (1 + np.cos(np.pi * (overlap - pos[:overlap]) / overlap))
                            pos3D_roi['weight'][:, :, :overlap] *= w[None, None, :]
                        if k < n_z - 1 and axis == 2:  # 下面重叠
                            w = 0.5 * (1 + np.cos(np.pi * pos[-overlap:] / overlap))
                            pos3D_roi['weight'][:, :, -overlap:] *= w[None, None, :]

                window_list.append((img3D_roi, pos3D_roi))

    return window_list


def finetune_model_predict3D(
        args,
        img3D,
        sam_model_tune,
        device="cuda",
        click_method="no_gt",
        num_clicks=10,
        prev_masks=None,
):
    try:
        print("开始预处理图像...")
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
        img3D = img3D.unsqueeze(dim=1)

        click_points = []
        click_labels = []
        pred_list = []

        if prev_masks is None:
            prev_masks = torch.zeros_like(img3D).to(device)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4),
        )

        print("图像形状:", img3D.shape)
        print("开始提取图像特征...")
        with torch.no_grad():
            try:
                image_embedding = sam_model_tune.image_encoder(
                    img3D.to(device)
                )
                print("图像特征形状:", image_embedding.shape)
            except Exception as e:
                print(f"图像编码器错误: {str(e)}")
                raise

        print(f"开始生成 {num_clicks} 个点击点...")
        for click_idx in range(num_clicks):
            try:
                with torch.no_grad():
                    print(f"\n处理第 {click_idx + 1} 个点击...")

                    # 生成点击点
                    batch_points, batch_labels = CLICK_METHODS[click_method](
                        prev_masks.to(device), img3D.to(device), 170
                    )
                    print(f"生成的点坐标: {batch_points}")
                    print(f"生成的点标签: {batch_labels}")

                    points_co = torch.cat(batch_points, dim=0).to(device)
                    points_la = torch.cat(batch_labels, dim=0).to(device)

                    click_points.append(points_co)
                    click_labels.append(points_la)

                    points_input = points_co
                    labels_input = points_la

                    print("生成特征嵌入...")
                    sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                        points=[points_input, labels_input],
                        boxes=None,
                        masks=low_res_masks.to(device),
                    )

                    print("解码掩码...")
                    low_res_masks, _ = sam_model_tune.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    prev_masks = F.interpolate(
                        low_res_masks,
                        size=img3D.shape[-3:],
                        mode="trilinear",
                        align_corners=False,
                    )

                    print("生成分割掩码...")
                    medsam_seg_prob = torch.sigmoid(prev_masks)
                    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                    pred_list.append(medsam_seg)
                    print(f"掩码 {click_idx + 1} 中的非零像素数量: {np.count_nonzero(medsam_seg)}")

            except Exception as e:
                print(f"处理点击 {click_idx + 1} 时出错: {str(e)}")
                raise

        return pred_list, click_points, click_labels

    except Exception as e:
        print(f"finetune_model_predict3D 发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        raise


def auto(args: Any, image3D: torch.Tensor, meta_info: Dict[str, Any], device: str, output_path: str) -> Dict[str, Any]:
    """自动模式：自动生成点进行分割"""
    try:
        # 获取保存路径
        vis_root, pred_path, img_name = get_save_paths(meta_info, args)
        print(f"保存路径: {pred_path}")

        # 初始化结果字典
        result = init_result_dict(pred_path)

        if args.skip_existing_pred and osp.exists(pred_path):
            print(f"跳过已存在的预测结果: {pred_path}")
            result["pred_path"] = pred_path
            return result

        # 设置滑动窗口
        print("设置滑动窗口...")
        pred3D_full_dict, sliding_window_list = setup_sliding_window(image3D, args)
        print(f"创建了 {len(sliding_window_list)} 个窗口")

        # 加载模型
        print("加载模型...")
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if os.path.exists(args.checkpoint_path):
            print(f"加载模型权重: {args.checkpoint_path}")
            model_dict = torch.load(args.checkpoint_path, map_location=device)
            sam_model_tune.load_state_dict(model_dict["model_state_dict"])
        else:
            print(f"错误：找不到模型权重文件: {args.checkpoint_path}")
            return None

        # 创建累积权重和预测的数组
        print("初始化累积数组...")
        weight_sum = {i: np.zeros_like(image3D.squeeze().numpy(), dtype=np.float32) for i in range(args.num_clicks)}
        pred_sum = {i: np.zeros_like(image3D.squeeze().numpy(), dtype=np.float32) for i in range(args.num_clicks)}

        print(f"\n开始处理 {len(sliding_window_list)} 个窗口...")
        # 处理每个窗口
        for window_idx, (image3D_window, pos3D) in enumerate(sliding_window_list):
            try:
                print(f"\n处理窗口 {window_idx + 1}/{len(sliding_window_list)}")
                print(f"窗口大小: {image3D_window.shape}")

                seg_mask_list, points, labels = finetune_model_predict3D(
                    args,
                    image3D_window,
                    sam_model_tune,
                    device=device,
                    click_method=args.point_method,
                    num_clicks=args.num_clicks,
                    prev_masks=None,
                )

                print(f"生成了 {len(seg_mask_list)} 个掩码")

                # 处理ROI区域
                ori_roi = pos3D["ori_roi"]
                print(f"处理ROI区域: {ori_roi}")

                for idx, seg_mask in enumerate(seg_mask_list):
                    try:
                        # 获取当前窗口的权重并转换为float32
                        weight = pos3D["weight"].astype(np.float32)
                        # 确保seg_mask也是float32类型
                        seg_mask = seg_mask.astype(np.float32)

                        # 将预测结果和权重累加到对应位置
                        pred_sum[idx][
                            ori_roi[0]:ori_roi[1],
                            ori_roi[2]:ori_roi[3],
                            ori_roi[4]:ori_roi[5]
                        ] += seg_mask * weight

                        weight_sum[idx][
                            ori_roi[0]:ori_roi[1],
                            ori_roi[2]:ori_roi[3],
                            ori_roi[4]:ori_roi[5]
                        ] += weight

                        print(f"窗口 {window_idx + 1}, 掩码 {idx + 1}: ROI区域 = {ori_roi}")
                        print(f"当前掩码中的非零像素数量: {np.count_nonzero(seg_mask)}")
                        print(f"累积权重中的非零像素数量: {np.count_nonzero(weight_sum[idx])}")

                    except Exception as e:
                        print(f"处理掩码 {idx + 1} 时出错: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        continue

            except Exception as e:
                print(f"处理窗口 {window_idx + 1} 时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        print("\n开始融合预测结果...")
        # 融合所有窗口的预测结果
        for idx in range(args.num_clicks):
            try:
                # 避免除零
                mask = weight_sum[idx] > 0
                # 创建一个新的全零数组
                final_pred = np.zeros_like(image3D.squeeze().numpy(), dtype=np.uint8)
                # 只在有效区域进行除法运算
                if mask.any():
                    final_pred[mask] = (pred_sum[idx][mask] / weight_sum[idx][mask] > 0.5).astype(np.uint8)
                pred3D_full_dict[idx] = final_pred

                print(f"掩码 {idx + 1} 中的非零像素数量: {np.count_nonzero(final_pred)}")

                # 保存结果
                result["masks"].append(pred3D_full_dict[idx])
                current_result = convert_ndarray(result)
                with open(output_path, "r", encoding='utf-8') as f:
                    result_list = json.load(f)
                result_list.append(current_result)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(result_list, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"处理最终掩码 {idx + 1} 时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        # 计算点的偏移
        print("\n处理点击点坐标...")
        padding_params = sliding_window_list[-1][-1]["padding_params"]
        cropping_params = sliding_window_list[-1][-1]["cropping_params"]
        point_offset = np.array([
            cropping_params[0] - padding_params[0],
            cropping_params[2] - padding_params[2],
            cropping_params[4] - padding_params[4],
        ])

        # 调整点的坐标
        points = [p.cpu().numpy() + point_offset for p in points]
        labels = [l.cpu().numpy() for l in labels]

        # 添加点和标签并保存
        result["points"] = points
        result["labels"] = labels
        current_result = convert_ndarray(result)
        with open(output_path, "r", encoding='utf-8') as f:
            result_list = json.load(f)
        result_list.append(current_result)
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)

        # 保存结果
        print("\n保存最终结果...")
        save_results(result, image3D, meta_info, vis_root, img_name,
                    points, labels, pred3D_full_dict, args)

        print(f"Done! 结果已保存到 {pred_path}")
        return result
        
    except Exception as e:
        print(f"auto函数发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        raise


def run_inference(params):
    """Main inference function."""
    args = init_args(params)
    model = load_model(args)

    # 创建输出目录
    output_dir = f"/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/output_json/{user_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"output.json"
    output_path = os.path.join(output_dir, output_filename)

    # 创建或清空JSON文件
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # Setup dataset
    all_dataset_paths = glob(join(args.test_data_path, "*"))  # 下一级
    all_dataset_paths = list(filter(osp.isdir, all_dataset_paths))
    print("len of data:", len(all_dataset_paths))

    dataset = Dataset_Union_ALL_Infer(
        paths=all_dataset_paths,
        data_type="",
        transform=tio.ToCanonical(),
        get_all_meta_info=True,  # 设置该参数为true才能获取到meta_info
    )
    dataloader = DataLoader(dataset=dataset, sampler=None, batch_size=1, shuffle=False)

    for batch_data in tqdm(dataloader, desc="Processing images"):
        image3D, meta_info = batch_data
        print("img shape:", image3D.shape)
        if args.dot_list == None:  # 选择自动标
            print("auto")
            result = auto(args, image3D, meta_info, model.device, output_path)
            print("points:", result[
                "points"])  # points: [array([[[279, 217,  47]]]), array([[[247, 225,  69]]]), array([[[195, 308,  29]]])]
            print("labels:", result["labels"])  # labels: [array([[0]]), array([[0]]), array([[0]])]
            print("masks:", np.array(result["masks"]).shape)  # list:(3, 1, 1, 512, 512, 82)  (num_clicks,img_shape)
            print("pred_path:",
                  result["pred_path"])  # ./visualization/test_amos/pred/validation/CT/amos_0013_pred2.nii.gz
        else:
            print("self")
            result = process_3D_image(image3D, model, args.num_clicks, meta_info, args, output_path, dot_list=args.dot_list)
            print("points:", result[
                "points"])  # points: [array([[[279, 217,  47]]]), array([[[247, 225,  69]]]), array([[[195, 308,  29]]])]
            print("labels:", result["labels"])  # labels: [array([[0]]), array([[0]]), array([[0]])]
            print("pred_path:",
                  result["pred_path"])  # ./visualization/test_amos/pred/validation/CT/amos_0013_pred2.nii.gz
    print("\n处理完成！")
    print(f"所有结果已保存到: {output_path}")
    return output_path

def convert_ndarray(obj):
    """递归转换所有 numpy.ndarray 为 list"""
    if obj is None:
        return None  # 直接返回 None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # `numpy.ndarray` 转 `list`
    elif isinstance(obj, np.integer):
        return int(obj)  # `numpy.int64` → `int`
    elif isinstance(obj, np.floating):
        return float(obj)  # `numpy.float32` → `float`
    elif isinstance(obj, dict):
        return {key: convert_ndarray(value) for key, value in obj.items()}  # 递归字典
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]  # 递归列表
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray(item) for item in obj)  # 递归元组
    else:
        return obj

def reconstruct_mask(mask_info):
    """从压缩格式重建掩码"""
    if mask_info is None:
        return None
    mask = np.zeros(mask_info["shape"], dtype=np.uint8)
    indices = tuple(np.array(idx) for idx in mask_info["indices"])
    mask[indices] = mask_info["values"]
    return mask

if __name__ == "__main__":
    # Example usage
    # params = {
    #     "test_data_path": "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/data/validation",  # 该文件夹下必须还有两个文件夹，然后才是图片
    #     "checkpoint_path": "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SAM-Med3D/ckpt/sam_med3d_turbo.pth",
    #     "num_clicks": 3,
    #     "dot_list": [
    #         (100, 100, 50, 0),  # (x, y, z, label) 前景点
    #         (200, 150, 50, 0),  # 都只点击前景
    #         (150, 120, 50, 0),  # 前景点
    #     ]
    # }
    # 0表示前景 -1表示背景

    params = json.loads(sys.argv[1])
    # 将 dot_list 的数组转换为元组,java端没办法传元组
    if "dot_list" in params:
        params["dot_list"] = [tuple(point) for point in params["dot_list"]]
    user_id = sys.argv[2]
    json_file_name = sys.argv[3]
    print("Received params:", params)
    print("User ID:", user_id)
    print("JSON file name:", json_file_name)

    result_path=run_inference(params)  # todo:返回值
