
import torch
from torch import  nn

import numpy as np
from mmaction.models import RECOGNIZERS
from mmaction.models.recognizers import BaseRecognizer, Recognizer3D
from mmaction.models.builder import build_head
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_world_size()


@RECOGNIZERS.register_module()
class SimilarityRecognizer3D(BaseRecognizer):
    def __init__(self, backbone, sim_head=None,max_batch_size=6, cls_head=None, neck=None,   train_cfg=None, test_cfg=None):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg)
        self.sim_head = build_head(sim_head) if sim_head is not None else nn.Identity()
        self.max_batch_size = max_batch_size
        self.fp16_enabled = False
        self.cos_similar = torch.nn.CosineSimilarity()
        self.relu = torch.nn.ReLU()

    def extract_feat(self, imgs):
        max_bs = self.max_batch_size
        n_imgs = imgs.size(0)
        st = 0
        all_feats = []

        if n_imgs < max_bs:
            feats = self.backbone(imgs)
            feats = self.sim_head(feats)
            return feats

        while st < n_imgs:
            cur_imgs = imgs[st : st+max_bs]
            feats = self.backbone(cur_imgs)
            feats = self.sim_head(feats)
            all_feats.append(feats)
            st += max_bs
        all_feats = torch.cat(all_feats, dim=0)
        return all_feats

    def forward_train(self, imgs, labels, video_id=None, frame_inds=None,**kwargs):
        """Defines the computation performed at every call when training."""
        assert video_id is not None
        c = imgs.size(1)
        #print("imgs :{}".format(imgs.size()))
        assert c  == 2 or c == 3
        bs = imgs.size(0)
        
        if video_id is None:
            sim_labels = torch.arange(0, bs).to(imgs.device)  + (get_rank() + 1) *  get_world_size()
        else:
            sim_labels = video_id
        sim_labels = sim_labels.reshape(-1, 1).repeat(1, c).reshape(-1,)
        
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        
        losses = dict()
        x = self.extract_feat(imgs)
    
        embeddings = self.sim_head(x)
        loss_sim = self.sim_head.loss(embeddings, sim_labels, **kwargs)
        losses.update(loss_sim)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                emb = self.extract_feat(batch_imgs)
                feats.append(emb)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                embs = tuple(feat)
            else:
                embs = torch.cat(feats)

        else:
            embs = self.extract_feat(imgs)

        feat = embs
        feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
            feat.size())
        if feat_dim != 2:  # 3D-CNN architecture
            feat = feat.flatten(2).mean(2)
            if num_segs == 1:
                feat = feat.reshape((batches, -1))
            else:
                feat = feat.reshape((batches, num_segs, -1))
        else:
            feat = feat.reshape((batches, num_segs, -1))
            if num_segs == 1:
                feat = feat[:, 0, :]
        return feat

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).detach().cpu().numpy()

    def forward_gradcam(self, imgs):
        return self.forward_test(imgs)
    
    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        # assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        feats = self.extract_feat(imgs)
        return (feats, )