import logging
import torch




def load_checkpoint(train_config, path, map_location='cuda'):
    # logging.info('load_checkpoint for------->')
    model: torch.nn.Module = make_training_model(train_config)
    model_dict = model.state_dict()

    pretrained_dict = torch.load(path, map_location=map_location)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.on_load_checkpoint(pretrained_dict)
    return model


if __name__ == '__main__':
    path = '/media/lab225/diskA/LMK/models/DiffIR_strip_changev2_4/last.ckpt'

