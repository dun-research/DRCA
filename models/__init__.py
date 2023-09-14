# register modules
from models.rat import (ResolutionAlignDividedTemporalAttentionWithNorm,
                        ResolutionAlignDividedSpatialAttentionWithNorm, 
                        ResolutionAlignFFN)
from models.rat_layers import ResolutionAlignTransformerLayerSequence
from models.drca import DRCA
from models.cls_head import ClsHead