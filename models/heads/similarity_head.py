

import torch.nn as nn

from models.losses.metric_loss import MetricLoss
from mmaction.models.heads import BaseHead
from mmaction.models.builder import HEADS
from collections import OrderedDict


@HEADS.register_module()
class SimilarityHead(BaseHead):
    def __init__(self, encoder=None, 
                       metric_loss=dict(type="MultiSimilarityLoss",
                                               distance="Cosine",
                                               miner_name="MultiSimilarityMiner"),
                        ):
        super().__init__(num_classes=1, in_channels=encoder['input_channels'])
        
        encoder_type = encoder.pop("type")
        out_channel = encoder['out_channels']
        if encoder_type == "FC":
            model = nn.Linear(encoder['input_channels'], encoder['out_channels'])
        elif encoder_type == "FCBN":
            model = nn.Sequential(OrderedDict([
                    ("conv1",nn.Linear(encoder['input_channels'], encoder['out_channels'])),
                    ("bn1",nn.BatchNorm1d(encoder['input_channels']))])
                )
        elif encoder_type == "None":
            model = nn.Identity()
            out_channel = encoder['input_channels']
        else:
             model = None
        
        self.encoder = model
        print(f"sim_head.encoder = {model}" )
        if metric_loss.get('feat_dim', -1) <=0 :
            metric_loss['feat_dim'] = out_channel
        self.loss_sim = MetricLoss(**metric_loss)
        self.pool = nn.AdaptiveAvgPool3d([1,None, None])
        self.fp16_enabled = False
    
    def init_weights(self):
        self.encoder.init_weights()

    def forward(self, features, **kwargs):
        n_dim = len(features.size())
        if self.encoder is not None:
            assert n_dim in [2, 5]
            if n_dim == 5:
                features = self.pool(features).squeeze()
            
            features = self.encoder(features)
            return features
        else:
            raise NotImplemented

    def loss(self, embeddings, labels, **kwargs):
        #loss = self.loss_sim(embeddings, labels, **kwargs)
        n_dim = len(embeddings.size())
        # print("embeddings :{}".format(embeddings.size()))
        if n_dim > 2 :
            embeddings = embeddings.flatten(2).mean(dim=2)
        loss = self.loss_sim(embeddings, labels, **kwargs)

        return {"loss_sim": loss}



