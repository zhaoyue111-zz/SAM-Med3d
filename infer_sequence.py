"""
Run inference without label masks. Based on inference.py, and requires new click methods 
from updated utils/click_method.py. Check the new click method details for more information.

Author: Karson Chrispens
Date: 5/15/2024
"""
import gzip
import os
import os.path as osp

import sys
join = osp.join
import argparse
import json
import pickle
from collections import OrderedDict, defaultdict
from glob import glob
from itertools import product

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from utils.click_method import (
    get_next_click3D_torch_no_gt_naive,
    get_next_click3D_torch_no_gt,
)
from utils.data_loader import Dataset_Union_ALL_Infer

class Args:
    def __init__(self):
        # 设置所有默认参数
        self.test_data_path = "./data/validation"
        self.checkpoint_path = "./ckpt/sam_med3d_turbo.pth"
        self.output_dir = "./visualization"
        self.task_name = "test_amos"
        self.user_id = "1"
        self.dot_list=None
        self.skip_existing_pred = False
        self.save_image = True
        self.sliding_window = False
        self.image_size = 256
        self.crop_size = 128
        self.device = "cuda"
        self.model_type = "vit_b_ori"
        self.num_clicks = 5
        self.point_method = "no_gt"
        self.data_type = "infer"
        self.threshold = 0
        self.dim = 3
        self.split_idx = 0
        self.split_num = 1
        self.ft2d = False
        self.seed = 2025


click_methods = {
    "no_gt": get_next_click3D_torch_no_gt,
    "no_gt_naive": get_next_click3D_torch_no_gt_naive,
}

def finetune_model_predict3D(
    img3D,
    sam_model_tune,
    device="cuda",
    click_method="no_gt",
    num_clicks=10,
    prev_masks=None,
    dot_list=None
):
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

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(
            img3D.to(device)
        )  # (1, 384, 16, 16, 16)

    print("dot_list:",dot_list)
    for click_idx in range(num_clicks):  # 不管是手动还是自动，都会传num_clicks
        with torch.no_grad():

            if dot_list is None:  # 自动
                batch_points, batch_labels = click_methods[click_method](
                    prev_masks.to(device), img3D.to(device), 170
                )  # default threshold is 170, showing that here

                points_co = torch.cat(batch_points, dim=0).to(device)
                points_la = torch.cat(batch_labels, dim=0).to(device)
                print("auto in finetune_model_predict3D,points_co:",points_co)  # tensor([[[62, 94, 82]]], device='cuda:0')
                print("points_la:",{points_la}) # tensor([[0]], device='cuda:0')
            else:  # 手动
                x, y, z, label = dot_list[click_idx]
                points_co = torch.tensor([[[x, y, z]]], dtype=torch.float).to(device)
                points_la = torch.tensor([[label]], dtype=torch.int64).to(device)
                print("manual in finetune_model_predict3D,points_co:", points_co)  # tensor([[[62, 94, 82]]], device='cuda:0')
                print("points_la:", {points_la})

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            # 获取点的嵌入表示
            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            # 解码生成掩码
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                multimask_output=False,
            )
            # 调整掩码大小
            prev_masks = F.interpolate(
                low_res_masks,
                size=img3D.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

            # 将掩码转换为二值图像
            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

    print("pred_list shape:",np.array(pred_list).shape)  # (3, 128, 128, 128)

    return pred_list, click_points, click_labels


# TODO: check if this works?
def pad_and_crop_with_sliding_window(img3D, crop_transform, offset_mode="center"):
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img3D.squeeze(0)),
    )
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    # cropping_params: (x_start, x_max-(x_start+roi_size), y_start, ...)
    # padding_params: (x_left_pad, x_right_pad, y_left_pad, ...)
    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)
    roi_shape = crop_transform.target_shape
    vol_bound = (0, img3D.shape[2], 0, img3D.shape[3], 0, img3D.shape[4])
    center_oob_ori_roi = (
        cropping_params[0] - padding_params[0],
        cropping_params[0] + roi_shape[0] - padding_params[0],
        cropping_params[2] - padding_params[2],
        cropping_params[2] + roi_shape[1] - padding_params[2],
        cropping_params[4] - padding_params[4],
        cropping_params[4] + roi_shape[2] - padding_params[4],
    )
    window_list = []
    offset_dict = {
        "rounded": list(product((-32, +32, 0), repeat=3)),
        "center": [(0, 0, 0)],
    }
    for offset in offset_dict[offset_mode]:
        # get the position in original volume~(allow out-of-bound) for current offset
        oob_ori_roi = (
            center_oob_ori_roi[0] + offset[0],
            center_oob_ori_roi[1] + offset[0],
            center_oob_ori_roi[2] + offset[1],
            center_oob_ori_roi[3] + offset[1],
            center_oob_ori_roi[4] + offset[2],
            center_oob_ori_roi[5] + offset[2],
        )
        # get corresponing padding params based on `vol_bound`
        padding_params = [0 for i in range(6)]
        for idx, (ori_pos, bound) in enumerate(zip(oob_ori_roi, vol_bound)):
            pad_val = 0
            if idx % 2 == 0 and ori_pos < bound:  # left bound
                pad_val = bound - ori_pos
            if idx % 2 == 1 and ori_pos > bound:
                pad_val = ori_pos - bound
            padding_params[idx] = pad_val
        # get corresponding crop params after padding
        cropping_params = (
            oob_ori_roi[0] + padding_params[0],
            vol_bound[1] - oob_ori_roi[1] + padding_params[1],
            oob_ori_roi[2] + padding_params[2],
            vol_bound[3] - oob_ori_roi[3] + padding_params[3],
            oob_ori_roi[4] + padding_params[4],
            vol_bound[5] - oob_ori_roi[5] + padding_params[5],
        )
        # pad and crop for the original subject
        pad_and_crop = tio.Compose(
            [
                tio.Pad(padding_params, padding_mode=crop_transform.padding_mode),
                tio.Crop(cropping_params),
            ]
        )
        subject_roi = pad_and_crop(subject)
        img3D_roi = subject_roi.image.data.clone().detach().unsqueeze(1)

        # collect all position information, and set correct roi for sliding-windows in
        # todo: get correct roi window of half because of the sliding
        windows_clip = [0 for i in range(6)]
        for i in range(3):
            if offset[i] < 0:
                windows_clip[2 * i] = 0
                windows_clip[2 * i + 1] = -(roi_shape[i] + offset[i])
            elif offset[i] > 0:
                windows_clip[2 * i] = roi_shape[i] - offset[i]
                windows_clip[2 * i + 1] = 0
        pos3D_roi = dict(
            padding_params=padding_params,
            cropping_params=cropping_params,
            ori_roi=(
                cropping_params[0] + windows_clip[0],
                cropping_params[0]
                + roi_shape[0]
                - padding_params[0]
                - padding_params[1]
                + windows_clip[1],
                cropping_params[2] + windows_clip[2],
                cropping_params[2]
                + roi_shape[1]
                - padding_params[2]
                - padding_params[3]
                + windows_clip[3],
                cropping_params[4] + windows_clip[4],
                cropping_params[4]
                + roi_shape[2]
                - padding_params[4]
                - padding_params[5]
                + windows_clip[5],
            ),
            pred_roi=(
                padding_params[0] + windows_clip[0],
                roi_shape[0] - padding_params[1] + windows_clip[1],
                padding_params[2] + windows_clip[2],
                roi_shape[1] - padding_params[3] + windows_clip[3],
                padding_params[4] + windows_clip[4],
                roi_shape[2] - padding_params[5] + windows_clip[5],
            ),
        )
        pred_roi = pos3D_roi["pred_roi"]

        # if((gt3D_roi[pred_roi[0]:pred_roi[1],pred_roi[2]:pred_roi[3],pred_roi[4]:pred_roi[5]]==0).all()):
        # print("skip empty window with offset", offset)
        #    continue

        window_list.append((img3D_roi, pos3D_roi))
    return window_list


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)

def get_save_paths(meta_info, args):
    """获取保存路径。

    Args:
        meta_info: 图像元信息
        args: 参数配置

    Returns:
        vis_root: 可视化结果根目录
        pred_path: 预测结果保存路径
        img_name: 图像名称
    """
    img_name = meta_info["image_path"][0]
    # 从路径中提取任务名称
    task_name = osp.basename(osp.dirname(img_name))
    # 构建用户专属目录下的任务目录
    mode = "manual" if args.dot_list is not None else "auto"
    vis_root = osp.join(args.pred_output_dir, task_name, mode)
    pred_path = osp.join(
        vis_root,
        osp.basename(img_name).replace(
            ".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"
        ),
    )
    return vis_root, pred_path, img_name

def unzipAndRemove() :
    last_click_idx = args.num_clicks - 1

    # 遍历目录下的所有nii.gz文件
    mode = "manual" if args.dot_list is not None else "auto"
    vis_root = osp.join(args.pred_output_dir, args.task_name, mode)

    for file in glob(join(vis_root, "*.nii.gz")):
        # 保留最后一个点的预测结果和带点的预测结果，以及原始图像
        if (f"_pred{last_click_idx}.nii.gz" in file or
                f"_pred{last_click_idx}_wPt.nii.gz" in file or
                "_img.nii.gz" in file):
            try:
                # 读取.nii.gz文件
                img = sitk.ReadImage(file)
                # 构造新的.nii文件名
                new_file = file.replace(".gz", "")
                # 使用UseCompression=False来保存未压缩的.nii文件
                g_file=gzip.GzipFile(file)
                open(new_file, "wb+").write(g_file.read())
                g_file.close()
                # sitk.WriteImage(img, new_file, useCompression=False)
                # 删除原始的.nii.gz文件
                os.remove(file)
                print(f"解压文件: {file} -> {new_file}")
            except Exception as e:
                print(f"解压文件失败 {file}: {str(e)}")
        else:
            # 删除其他文件
            try:
                os.remove(file)
                print(f"删除文件: {file}")
            except Exception as e:
                print(f"删除文件失败 {file}: {str(e)}")


if __name__ == "__main__":
    # 数据：data/validation/{user_id}/{taskname}/.nii.gz
    # 掩码：./visualization/{user_id}/pred/{taskname}/auto.pred_nii.gz
    #      ./visualization/{user_id}/pred/{taskname}/manual/.pred_nii.gz
    global args
    args = Args()
    args.num_clicks = int(sys.argv[1])  # 记得转换类型
    args.user_id = sys.argv[2]
    args.task_name = sys.argv[3]
    # 检查是否有第四个参数（dot_list）
    if len(sys.argv) > 4:  # 对于dot_list，不能判断len==4
        try:
            # 如果有第四个参数，尝试将其转换为点列表
            import ast
            args.dot_list = ast.literal_eval(sys.argv[4])
            print("dot_list转换成功", args.dot_list)
        except:
            # 如果转换失败，保持为 None
            args.dot_list = None
            print("dot_list转换失败",None)
    else:
        # 如果没有第四个参数，保持为 None
        args.dot_list = None

    # todo：用户输入  直接运行python程序时可以不带双引号，可以带空格，但是java调用必须带双引号，不能有空格
    # args.dot_list = [
    #     [65, 95, 81, 0],  # [x, y, z, label]
    #     [90, 73, 119, 0],
    #     [70, 85, 15, 0]
    # ]
    print("args.num_clicks:", args.num_clicks)   # 不管穿不穿点数，该属性必须穿
    print("user_id:", args.user_id)
    print("args.task_name:", args.task_name)
    print("args.dot_list:",args.dot_list)

    SEED = args.seed
    print("set seed as", SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.init()

    # 修改输出目录为用户专属目录
    args.output_dir = join(args.output_dir, args.user_id)
    args.pred_output_dir = join(args.output_dir, "pred")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pred_output_dir, exist_ok=True)
    args.save_name = join(args.output_dir, "dice.py")
    print("output_dir set to", args.output_dir)

    # 读取test_data_path下的所有文件
    args.test_data_path = join(args.test_data_path, args.user_id,args.task_name)
    nii_files = glob(join(args.test_data_path, "*.nii.gz"))
    print(f"找到 {len(nii_files)} 个.nii.gz文件")

    crop_transform = tio.CropOrPad(
        target_shape=(args.crop_size, args.crop_size, args.crop_size)
    )

    infer_transform = [
        tio.ToCanonical()
    ]

    test_dataset = Dataset_Union_ALL_Infer(
        paths=[args.test_data_path],
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
        get_all_meta_info=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=False
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    if args.dim == 3:
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(
            device
        )
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict["model_state_dict"]
            sam_model_tune.load_state_dict(state_dict)
    else:
        raise NotImplementedError(
            "this scipts is designed for 3D sliding-window inference, not support other dims"
        )

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    for batch_data in tqdm(test_dataloader):
        image3D, meta_info = batch_data
        print("meta_info:",meta_info)
        img_name = meta_info["image_path"][0]

        modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
        dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
        vis_root, pred_path, img_name = get_save_paths(meta_info, args)

        """ inference """
        if args.skip_existing_pred and osp.exists(pred_path):
            pass  # if the pred existed, skip the inference
        else:
            image3D_full = image3D
            pred3D_full_dict = {
                click_idx: torch.zeros_like(image3D_full).numpy()
                for click_idx in range(args.num_clicks)
            }
            offset_mode = "center" if (not args.sliding_window) else "rounded"
            sliding_window_list = pad_and_crop_with_sliding_window(
                image3D_full, crop_transform, offset_mode=offset_mode
            )
            for image3D, pos3D in sliding_window_list:
                seg_mask_list, points, labels = finetune_model_predict3D(
                    image3D,
                    sam_model_tune,
                    device=device,
                    click_method=args.point_method,
                    num_clicks=args.num_clicks,
                    prev_masks=None,
                    dot_list=args.dot_list
                )
                print("points",points)
                ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]
                for idx, seg_mask in enumerate(seg_mask_list):
                    seg_mask_roi = seg_mask[
                        ...,
                        pred_roi[0] : pred_roi[1],
                        pred_roi[2] : pred_roi[3],
                        pred_roi[4] : pred_roi[5],
                    ]
                    pred3D_full_dict[idx][
                        ...,
                        ori_roi[0] : ori_roi[1],
                        ori_roi[2] : ori_roi[3],
                        ori_roi[4] : ori_roi[5],
                    ] = seg_mask_roi

            os.makedirs(vis_root, exist_ok=True)
            padding_params = sliding_window_list[-1][-1]["padding_params"]
            cropping_params = sliding_window_list[-1][-1]["cropping_params"]
            # print(padding_params, cropping_params)
            point_offset = np.array(
                [
                    cropping_params[0] - padding_params[0],
                    cropping_params[2] - padding_params[2],
                    cropping_params[4] - padding_params[4],
                ]
            )
            points = [p.cpu().numpy() + point_offset for p in points]
            labels = [l.cpu().numpy() for l in labels]
            pt_info = dict(points=points, labels=labels)
            # print("save to", osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_pred.nii.gz")))
            pt_path = osp.join(
                vis_root, osp.basename(img_name).replace(".nii.gz", "_pt.pkl")
            )
            pickle.dump(pt_info, open(pt_path, "wb"))

            if args.save_image:
                save_numpy_to_nifti(
                    image3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(".nii.gz", f"_img.nii.gz"),
                    ),
                    meta_info,
                )
            for idx, pred3D_full in pred3D_full_dict.items():
                save_numpy_to_nifti(
                    pred3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(".nii.gz", f"_pred{idx}.nii.gz"),
                    ),
                    meta_info,
                )
                radius = 2
                for pt in points[: idx + 1]:
                    x, y, z = pt[0, 0]
                    pred3D_full[
                        ...,
                        int(x - radius): int(x + radius),
                        int(y - radius): int(y + radius),
                        int(z - radius): int(z + radius),
                    ] = 10
                save_numpy_to_nifti(
                    pred3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(
                            ".nii.gz", f"_pred{idx}_wPt.nii.gz"
                        ),
                    ),
                    meta_info,
                )

    unzipAndRemove()

    print("Done")
