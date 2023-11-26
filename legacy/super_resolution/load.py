import torch
from legacy.super_resolution.vapsr import vapsr


def load(model_path, scale):
    if scale == 2:
        model = vapsr(num_in_ch=3, num_out_ch=3, num_feat=48, d_atten=64, num_block=20, scale=2)

    elif scale == 4:
        model = vapsr(num_in_ch=3, num_out_ch=3, num_feat=48, d_atten=64, num_block=21, scale=4)

    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model.cpu()
    return model
