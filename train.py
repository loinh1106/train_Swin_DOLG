import apex
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import argparse
import random
import time
import os
import glob
import re
import math

from torch.backends import cudnn
from tqdm import tqdm as tqdm
from apex.parallel import DistributedDataParallel
from apex import amp

from configs.config import init_config
from model.DOLG import DOLG
from model.hybrid_swin_transformer import SwinTransformer
from model.loss import *
from utils.util import global_average_precision_score, GradualWarmupSchedulerV2
from data_loader.dataset import LandmarkDataset, get_df, get_transforms
from pathlib import Path

import torch 
import torch.nn.functional as F 
from torch import nn 
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class Mish_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        
        grad_hx = i.sigmoid()

        grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 
        
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Mish initialized")
        pass
    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)

class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=5, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False, gc_loc=True
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        #self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        #loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class ModelIRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(ModelIRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            print("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]
        
        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1
        
        return [lr for _ in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return self.base_lrs
    
    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) / 
                  self.lr_ramp_ep * self.last_epoch + 
                  self.lr_start)
        
        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
            
        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay**
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) + 
                  self.lr_min)
        return lr

def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x

def replace_activations(model, existing_layer, new_layer):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--trainCSVPath', type=str, required=True)
    parser.add_argument('--valCSVPath', type=str, required=True)
    parser.add_argument('--checkpoint', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--use_central_gradient', action='store_true')
    parser.add_argument('--use_mish', action='store_true')
    parser.add_argument('--loss_name', type=str, default="arcface_dynamicmargin")
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if not cfg['train']['use_amp']:
            _, logits_m = model(data)
            loss = loss_fn(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            _, logits_m = model(data)
            
            loss = loss_fn(logits_m, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)

        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    train_loss = np.mean(train_loss)
    return train_loss


def val_epoch(model, valid_loader, loss_fn, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            _,logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = loss_fn(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=
                  0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls,
                                                    conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m

def get_loader(cfg, df):
    sampler = torch.utils.data.distributed.DistributedSampler(df)
    loader = torch.utils.data.DataLoader(df, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                                shuffle=sampler is None, sampler=sampler, drop_last=True)
    return sampler, loader

# loss func
def make_loss(loss_name,cfg, margins, out_dim):
    if loss_name == "arcface_dynamicmargin":
        loss_fn = ArcFaceLossAdaptiveMargin(margins=margins, s=cfg['arcface_s'], out_dim = out_dim)
    elif loss_name=="cosface":
        loss_fn = Cosface(out_features = cfg['batch_size'], in_features=out_dim, m=torch.FloatTensor(margins).cuda())
    elif loss_name == "CE_smooth_loss":
        loss_fn = CrossEntropyLossWithLabelSmoothing(n_dim=out_dim)
        # loss_fn = LabelSmoothingCrossEntropy()
    elif loss_name == "smooth_CE_loss":
        loss_fn = LabelSmoothingCrossEntropy()
    elif loss_name == "circleloss":
        # loss_fn = CircleLoss(m=0.25, emdsize=2, class_num=32, gamma=64)
        loss_fn = CircleLoss(m=0.25, gamma=64)
    elif loss_name == "centerloss":
        loss_fn = CenterLoss(num_classes=out_dim, feat_dim=cfg['batch_size'], use_gpu=True)
    return loss_fn

def train(cfg, args):

    # get dataframe
    df_train, out_dim = get_df(args.trainCSVPath)
    df_val, _ = get_df(args.valCSVPath)

    # get adaptive margin
    tmp = np.sqrt(
        1 / np.sqrt(df_train['id_encode'].value_counts().sort_index().values))
    alpha = 1e-6
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min() + alpha) * 0.45 + 0.05

    # get augmentations (Resize and Normalize)
    transforms_train, transforms_val = get_transforms(cfg['train']['image_size'])

    dataset_train = LandmarkDataset(df_train, 'train', transform=transforms_train)
    dataset_val = LandmarkDataset(df_val, 'val', transform=transforms_val)


    model = DOLG(cfg).cuda() if args.model_name == "dolg" else SwinTransformer(cfg).cuda()
    model = apex.parallel.convert_syncbn_model(model)
    early_stopping = EarlyStopping(tolerance= 5, min_delta = 10)

    ####
    if args.use_mish:
        print("[INFO] Using Mish Activation")
        existing_layer = torch.nn.SiLU
        new_layer = Mish()
        model = replace_activations(model, existing_layer, new_layer) 

    ####
    if args.use_central_gradient:
        print("[INFO] Using Central Gradient")
        SCHEDULER_PARAMS = {
            "lr_start": 1e-5,
            "lr_max": 1e-5 * 32,
            "lr_min": 1e-6,
            "lr_ramp_ep": 5,
            "lr_sus_ep": 0,
            "lr_decay": 0.8,
        }

        optimizer = Ranger(model.parameters(), lr = SCHEDULER_PARAMS['lr_start'])
        scheduler = ModelIRScheduler(optimizer,**SCHEDULER_PARAMS)
    else:
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg['train']['init_lr'])
        if cfg['train']['use_amp']:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            
        # lr scheduler
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, cfg['train']['n_epochs']-1)
        scheduler_warmup = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    loss_fn = make_loss(args.loss_name, cfg['train'], margins, out_dim)

    best_fitness = 0.0
    # load pretrained
    if args.checkpoint:
        print('-------Load Checkpoint-------')
        cfg['train']['model_dir'] = '/'.join(args.checkpoint.split('/')[:-1])
        args.exist_ok = True
        checkpoint = torch.load(args.checkpoint,  map_location='cuda:{}'.format(cfg['train']['local_rank']))
        cfg['train']['start_from_epoch'] = checkpoint['epoch'] + 1
        best_fitness = checkpoint['best_fitness']
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith(
            'module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print('-------DONE-------')

    model = DistributedDataParallel(model, delay_allreduce=True)
   

    # Directories
    model_path = increment_path(Path(cfg['train']['model_dir']), exist_ok=args.exist_ok)
    os.makedirs(model_path, exist_ok=True)
    last = os.path.join(model_path, 'last.pth')
    best = os.path.join(model_path, 'best.pth')

     # train & valid loop
    gap_m_max = 0
    for epoch in range(cfg['train']['start_from_epoch'], cfg['train']['n_epochs']+1):
        print(time.ctime(), 'Epoch:', epoch)
        if args.use_central_gradient:
            scheduler.step()
        else:
            scheduler_warmup.step(epoch - 1)
        

        train_sampler, train_loader = get_loader(cfg['train'], dataset_train)
        train_sampler.set_epoch(epoch)

        val_sampler, val_loader = get_loader(cfg['val'], dataset_val)
        val_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, acc_m, gap_m = val_epoch(model, val_loader, loss_fn)
        
        

        content = time.ctime() + ' ' + \
                f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, \
                train loss: {(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
        print(content)

        if gap_m > best_fitness:
            best_fitness = gap_m

        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

        if best_fitness == gap_m:
            print(f"Save best epoch: {epoch}")
            torch.save(ckpt, best)
        if epoch % cfg['train']['save_per_epoch'] == 0:
            save_dir = os.path.join(model_path, 
                            "dolg_{}_{}.pth".format(cfg['train']['model_name'], epoch))
            print('gap_m_max ({:.6f} --> {:.6f}). Saving model to {}'.format(gap_m_max, gap_m, save_dir))
            torch.save(ckpt, save_dir)
            gap_m_max = gap_m
        
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print(f'Stop at epoch {epoch}')
            break


if __name__ == '__main__':

    args = parse_args()
    if args.config_name == None:
        assert "Wrong config_file.....!"

    cfg = init_config(args.config_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['train']['CUDA_VISIBLE_DEVICES']
    set_seed(0)
    
    if cfg['train']['CUDA_VISIBLE_DEVICES'] != '-1':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(cfg['train']['local_rank'])
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        cudnn.benchmark = True

    train(cfg, args)
