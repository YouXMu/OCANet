#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')  # 验证时，需要有模型的config。yaml
        with open(train_config_path,
                  'r') as f:  # 将 yaml.safe_load(f) 解析得到的Python数据结构转换为一个OmegaConf配置对象 并赋值给 train_config 变量
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True  # 禁用训练
        train_config.visualizer.kind = 'noop'  # 禁用可视化

        out_ext = predict_config.get('out_ext',
                                     '.png')  # 如果 predict_config 中存在 'out_ext' 这个键，那么它的值会被赋值给 out_ext 变量；如果不存在，则会使用默认值 '.png'

        checkpoint_path = os.path.join(predict_config.model.path,  # 获得拼接好的的模型保存的路径
                                       'models',
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')  # 加载模型参数
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):  # 保证代码的健壮性
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):  # 循环遍历数据集
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir,
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )  # 并根据配置信息构造修复后图像的输出文件名。
            os.makedirs(os.path.dirname(cur_out_fname),
                        exist_ok=True)  # 确保输出文件的目录存在。如果目录不存在，则创建它。exist_ok=True 表示如果目录已存在，则不会抛出错误。
            batch = default_collate([dataset[img_i]])  # 将单个数据样本组合成一个批次（尽管这里只处理一个样本）。
            if predict_config.get('refine',
                                  False):  # 如果 predict_config 字典中不存在 'refine' 这个键，那么 get 方法会返回其第二个参数作为默认值，即 False
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1  # 这行代码将掩码（mask）中的非零值转换为 1，零值保持不变
                    batch = model(batch)
                    cur_res = batch[predict_config.out_key][0].permute(1, 2,
                                                                       0).detach().cpu().numpy()  # [0] 表示取批次中的第一个样本（因为 batch 可能包含多个样本，但这里我们只处理一个） 将其转换为 [H, W, C]
                    unpad_to_size = batch.get('unpad_to_size',
                                              None)  # 如果模型或预处理步骤中使用了填充（padding），并且填充的大小被记录在了 batch 的 'unpad_to_size' 键中，那么这里会从修复后的图像中裁剪回原始大小
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype(
                'uint8')  # 使用 NumPy 的 clip 函数确保所有的像素值都在 [0, 255] 的范围内。任何超出这个范围的值都会被裁剪到边界值。 将处理后的数组的数据类型转换为无符号8位整数（uint8）
            # OpenCV（通常通过其 Python 接口 cv2 使用）默认使用 BGR（蓝绿红）颜色空间来存储图像，
            # 而许多其他库（包括 PyTorch 和 TensorFlow）通常使用 RGB（红绿蓝）颜色空间。
            # 这行代码使用 OpenCV 的 cvtColor 函数将图像从 RGB 颜色空间转换为 BGR 颜色空间。这是为了确保图像可以在 OpenCV 中正确显示和保存。
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)  # 用于将图像保存到磁盘上

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
