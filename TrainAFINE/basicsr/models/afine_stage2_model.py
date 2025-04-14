import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os
from copy import deepcopy
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch.nn.functional as F
import torch.nn as nn

from CLIP_ReturnFea import clip


@MODEL_REGISTRY.register()
class AFINEStage2Model(BaseModel):
    def __init__(self, opt):
        super(AFINEStage2Model, self).__init__(opt)

        # load pretrained CLIP
        pretrained_CLIP_path = opt['path_CLIP']['pretrain_CLIP_path']
        self.clip_model, _ = clip.load(pretrained_CLIP_path, device="cpu", jit = False)


        if opt['path_CLIP'].get('mode', 'Original') == 'finetune':
            print(f"Load my finetune")
            checkpoint = torch.load(opt['path_CLIP']['finetune_CLIP_path'], map_location = 'cpu')
            self.clip_model.load_state_dict(checkpoint)


        self.clip_model = self.clip_model.to(self.device)

        # define network dhead
        self.net_dhead = build_network(opt['network_dhead'])
        self.net_dhead = self.model_to_device(self.net_dhead)
        self.print_network(self.net_dhead)

        # load pretrained dhead
        load_path = self.opt['path_dhead'].get('pretrain_network_dhead', None)
        if load_path is not None:
            param_key = self.opt['path_dhead'].get('param_key_dhead', 'params')
            self.load_network(self.net_dhead, load_path, self.opt['path_dhead'].get('strict_load_dhead', True), param_key)


        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_dhead.train()

        train_opt = self.opt['train']

        if train_opt.get('finetune_CLIP', False):
            self.clip_model.train()

        # define losses
        if train_opt.get('fidelity_fr12_opt'):
            self.cri_fidelity_fr12 = build_loss(train_opt['fidelity_fr12_opt']).to(self.device)
        else:
            self.cri_fidelity_fr12 = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimize net_dhead
        optim_type = train_opt['optim_dhead'].pop('type')
        self.optimizer_dhead = self.get_optimizer(optim_type, self.net_dhead.parameters(), **train_opt['optim_dhead'])
        self.optimizers.append(self.optimizer_dhead)

        if train_opt.get('finetune_CLIP', False):
            optim_type = train_opt['optim_clip'].pop('type')
            self.optimizer_clip = self.get_optimizer(optim_type, self.clip_model.parameters(), **train_opt['optim_clip'])
            self.optimizers.append(self.optimizer_clip)

    def feed_data(self, data):
        self.img1 = data['img1'].to(self.device)
        self.img2 = data['img2'].to(self.device)
        self.ref = data['ref'].to(self.device)
        self.gt_score_12 = data['gt_score_12'].to(self.device)
        self.gt_score_1ref = data['gt_score_1ref'].to(self.device)
        self.gt_score_2ref = data['gt_score_2ref'].to(self.device)
        self.ori_h = data['ori_h']
        self.ori_w = data['ori_w']

    def clip_processing(self, img1, img2, ref, finetune_clip = False):
        if not finetune_clip:
            with torch.no_grad():
                cls_img1, feat_img1 = self.clip_model.encode_image(img1)
                cls_img2, feat_img2 = self.clip_model.encode_image(img2)
                cls_imgref, feat_imgref = self.clip_model.encode_image(ref)
        else:
            cls_img1, feat_img1 = self.clip_model.encode_image(img1)
            cls_img2, feat_img2 = self.clip_model.encode_image(img2)
            cls_imgref, feat_imgref = self.clip_model.encode_image(ref)

        return cls_img1, cls_img2, cls_imgref, feat_img1, feat_img2, feat_imgref

    def fr_model_processing(self, img1, img2, ref, feat_img1, feat_img2, feat_imgref):
        normal_dist = torch.distributions.Normal(0, 1)


        u1ref = self.net_dhead(img1, ref, feat_img1, feat_imgref)
        u2ref = self.net_dhead(img2, ref, feat_img2, feat_imgref)


        arg12 = (u2ref - u1ref) / torch.sqrt(torch.tensor(2.0))
        p_hat_12 = normal_dist.cdf(arg12)



        return p_hat_12, u1ref, u2ref



    def uniqa_model_processing(self, img1, img2, ref, finetune_clip = False):
        cls_img1, cls_img2, cls_imgref, feat_img1, feat_img2, feat_imgref = self.clip_processing(img1, img2, ref, finetune_clip = finetune_clip)

        p_hat_12, u1ref, u2ref = self.fr_model_processing(img1, img2, ref, feat_img1, feat_img2, feat_imgref)


        return p_hat_12, u1ref, u2ref

    def optimize_parameters(self, current_iter):
        for name, parameter in self.clip_model.named_modules():
            if isinstance(parameter, nn.BatchNorm2d):
                parameter.track_running_stats = False
                parameter.affine = False
                parameter.requires_grad = False
                parameter.eval()

        self.optimizer_dhead.zero_grad()

        if self.opt['train'].get('finetune_CLIP', False):
            self.optimizer_clip.zero_grad()

        b,c,h,w = self.img1.shape

        l_total = 0
        loss_dict = OrderedDict()

        l_fidelity_fr12 = 0

        for i in range(b):
            b_img1 = self.img1[i].unsqueeze(0)
            b_img2 = self.img2[i].unsqueeze(0)
            b_ref = self.ref[i].unsqueeze(0)
            b_ori_h = self.ori_h[i]
            b_ori_w = self.ori_w[i]

            b_img1 = b_img1[:, :, :b_ori_h, :b_ori_w]
            b_img2 = b_img2[:, :, :b_ori_h, :b_ori_w]
            b_ref = b_ref[:, :, :b_ori_h, :b_ori_w]

            # print(f"b_img1 shape is {b_img1.shape}")

            b_gt_score_12 = self.gt_score_12[i].reshape(1)
            b_gt_score_1ref = self.gt_score_1ref[i].reshape(1)
            b_gt_score_2ref = self.gt_score_2ref[i].reshape(1)

            p_hat_12, u1ref, u2ref = self.uniqa_model_processing(b_img1, b_img2, b_ref, finetune_clip = self.opt['train'].get('finetune_CLIP', False))


            # fidelity loss
            if self.cri_fidelity_fr12:
                l_fidelity_fr12_b = self.cri_fidelity_fr12(p_hat_12, b_gt_score_12.unsqueeze(1))
                l_fidelity_fr12 += l_fidelity_fr12_b


        l_fidelity_fr12 = l_fidelity_fr12 / b


        loss_dict['l_fidelity_fr12'] = l_fidelity_fr12

        l_total += l_fidelity_fr12

        l_total.backward()
        self.optimizer_dhead.step()
        if self.opt['train'].get('finetune_CLIP', False):
            self.optimizer_clip.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        for name, parameter in self.clip_model.named_modules():
            if isinstance(parameter, nn.BatchNorm2d):
                parameter.track_running_stats = False
                parameter.affine = False
                parameter.requires_grad = False
                parameter.eval()

        self.net_dhead.eval()
        self.clip_model.eval()

        _,c,h,w = self.img1.shape
        if h % 32 != 0:
            pad_h = 32 - h % 32
        else:
            pad_h = 0

        if w % 32 != 0:
            pad_w = 32 - w % 32
        else:
            pad_w = 0

        img1 = F.interpolate(self.img1, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)
        img2 = F.interpolate(self.img2, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)
        ref = F.interpolate(self.ref, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)

        with torch.no_grad():
            self.p_hat_12, self.u1ref, self.u2ref = self.uniqa_model_processing(img1, img2, ref, finetune_clip = False)
        self.net_dhead.train()
        if self.is_train and self.opt['train'].get('finetune_CLIP', False):
            self.clip_model.train()

    def feed_val_data(self, data):
        self.img1 = data['img1'].to(self.device)
        self.img2 = data['img2'].to(self.device)
        self.ref = data['ref'].to(self.device)
        self.gt_score = data['gt_score'].to(self.device)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):

            self.feed_val_data(val_data)
            self.test()

            metric_data['u1ref_all'] = self.u1ref.flatten().item()
            metric_data['u2ref_all'] = self.u2ref.flatten().item()
            metric_data['gt_score'] = self.gt_score.flatten().item()

            # print(f"u1ref_all is {metric_data['u1ref_all']}, u2ref_all is {metric_data['u2ref_all']}")
            # tentative for out of GPU memory
            del self.img1
            del self.img2
            del self.ref
            del self.gt_score
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
        if use_pbar:
            pbar.close()

        logger = get_root_logger()
        if with_metrics:
            for metric in self.metric_results.keys():
                logger.info(f"For dataset {dataset_name}, {self.metric_results[metric]} pairs are predicted correct.\n")
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\t Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save_clip(self, current_iter, net_label, model):
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        if self.opt['world_size'] > 1:
            torch.save(model.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)


    def save(self, epoch, current_iter):
        self.save_network(self.net_dhead, 'net_dhead', current_iter)
        if self.opt['train'].get('finetune_CLIP', False):
            self.save_clip(current_iter = current_iter, net_label = 'clip_model', model=self.clip_model)
        self.save_training_state(epoch, current_iter)
