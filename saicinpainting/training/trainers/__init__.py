import logging
import torch
from saicinpainting.training.trainers.defaultS1 import DefaultInpaintingTrainingModule
from saicinpainting.training.trainers.defaultS2 import SDefaultInpaintingTrainingModule

def get_training_model_class(kind):
    if kind == 'defaultS1':
        return DefaultInpaintingTrainingModule
    elif kind == 'defaultS2':
        return SDefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind

    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    # cls = cls.load_from_checkpoint(checkpoint_path=config.training_model.checkpoint)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', tag=True, strict=False):

    if tag:   #  预加载模型，加快当前模型的训练的收敛速度
        logging.info('load_checkpoint for-------> quickly convergence')
        model: torch.nn.Module = make_training_model(train_config)
        model_dict = model.state_dict()

        new = {}
        for k, v in model_dict.items():
            if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
                new[k] = v
        pretrained_dict = torch.load(path, map_location=map_location)
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in new}

        model.load_state_dict(pretrained_dict, strict=False)
        model.on_load_checkpoint(pretrained_dict)


    else:   # 测试时加载模型参数

        logging.info('load_checkpoint for-------> resume training ')
        model: torch.nn.Module = make_training_model(train_config)
        state = torch.load(path, map_location=map_location)
        model.load_state_dict(state['state_dict'], strict=strict)
        # model.load_from_checkpoint(path)
        model.on_load_checkpoint(state)

    return model


# def load_checkpoint(train_config, path, map_location='cuda'):
#     logging.info('load_checkpoint for------->')
#     model: torch.nn.Module = make_training_model(train_config)
#     model_dict = model.state_dict()
#
#     new = {}
#     for k,v in model_dict.items():
#         if "evaluator" not in k and "loss" not in k and "discriminator" not in k :
#             new[k] = v
#
#     # for k,v in model_dict.items():
#     #     print(k)
#     # crt = set( model_dict.keys() )
#     # for v in sorted(list(crt) ):
#     #     print(f' {v}')
#
#     pretrained_dict = torch.load(path, map_location=map_location)
#     pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in new}
#     # for k,v in pretrained_dict.items():
#     #     print(k)
#
#     # model_dict.update(pretrained_dict)
#     # for k,v in model_dict.items():
#     #     print(k)
#     model.load_state_dict(pretrained_dict,strict=False)
#     model.on_load_checkpoint(pretrained_dict)
#     return model



