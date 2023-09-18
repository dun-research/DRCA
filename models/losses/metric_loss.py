

from numpy import empty
import numpy as np
import torch
import torch.nn as nn
from mmaction.models.builder import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from pytorch_metric_learning import miners, losses, distances
from pytorch_metric_learning.utils import distributed as pml_dist

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch.distributed as dist
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@LOSSES.register_module()
class MetricLoss(BaseWeightedLoss):
    def __init__(self,  loss_weight=1, 
                        type="TripletMarginLoss",
                        miner_name="Empty",
                        margin=0.3, 
                        cross_batch_memory=False,
                        memory_size=512,
                        feat_dim=-1,
                        copy_ref_label=False,
                        distributed=False,
                        distance="Cosine"):
        super().__init__(loss_weight)
        if distance == "Cosine":
            dist_func = distances.CosineSimilarity()
        elif distance == "L2":
            dist_func = distances.LpDistance(p=2, power=2)
        elif distance == "Empty":
            dist_func = None
        else:
            raise NotImplemented("unsupported metric distance:{}".format(distance)) 

        if miner_name == "Empty":
            miner = None
        elif miner_name == "TripletMarginMiner":
            miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard", distance=dist_func)
        elif miner_name == "MultiSimilarityMiner":
            miner = miners.MultiSimilarityMiner(epsilon=margin, distance=dist_func)
        else:
            raise NotImplementedError("unsupported miner: {}".format(miner_name))               

        if type == "TripletMarginLoss":
            metric_loss = losses.TripletMarginLoss(margin=margin, distance=dist_func)
        elif type == "MultiSimilarityLoss":
            metric_loss = losses.MultiSimilarityLoss(distance=dist_func)
        else:
            raise NotImplementedError("unsupported metric loss:{}".format(type)) 
        
        efficient=False
        if cross_batch_memory:
            metric_loss = losses.CrossBatchMemory(metric_loss, embedding_size=feat_dim, memory_size=memory_size)
        self.distributed = distributed
        
        if distributed:
            metric_loss = pml_dist.DistributedLossWrapper(metric_loss,efficient=efficient)
            if miner is not None:
                miner = pml_dist.DistributedMinerWrapper(miner,efficient=efficient)
        self.distance = dist_func
        self.metric_loss = metric_loss
        self.miner = miner
        self.fp16_enabled = True

    def _forward(self, embeddings, targets, enqueue_idx=None):
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        labels = targets.reshape(-1,)
        loss = self.metric_loss(embeddings, labels.reshape(-1), indices_tuple=None,)
        return loss



