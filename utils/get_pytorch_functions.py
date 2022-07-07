import torch.nn.functional as F
import torch.optim as optim


# inspired by https://github.com/ncullen93/torchsample/blob/1f328d1ea3ef533c8c0c4097ed4a3fa16d784ba4/torchsample/modules/_utils.py#L72
def get_loss_function(loss):
    dir_f = dir(F)
    loss_fns = [d.lower() for d in dir_f]
    if isinstance(loss, str):
        try:
            str_idx = loss_fns.index(loss.lower())
        except ValueError as e:
            raise ValueError('Invalid loss string input - must match pytorch function. \n '
                             f'Error: {e}')
        return getattr(F, dir(F)[str_idx])

    # can also pass a function
    elif callable(loss):
        return loss
    else:
        raise ValueError('Invalid loss input')


# https://github.com/ncullen93/torchsample/blob/1f328d1ea3ef533c8c0c4097ed4a3fa16d784ba4/torchsample/modules/_utils.py#L93
def get_optimizer(optimizer):
    dir_optim = dir(optim)
    opts = [o.lower() for o in dir_optim]
    if isinstance(optimizer, str):
        try:
            str_idx = opts.index(optimizer.lower())
        except:
            raise ValueError('Invalid optimizer string input - must match pytorch function.')
        return getattr(optim, dir_optim[str_idx])
    # can pass a valid pytorch optimizer function, too
    elif hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        return optimizer
    else:
        raise ValueError('Invalid optimizer input')