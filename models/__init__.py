# register modules
from models.modules.reso_align_transformer import (ResolutionAlignDividedTemporalAttentionWithNorm,
                        ResolutionAlignDividedSpatialAttentionWithNorm, 
                        ResolutionAlignFFN)
from models.modules.reso_align_layers import ResolutionAlignTransformerLayerSequence
from models.backbones.drca import DRCA

from models.heads.cls_head import ClsHead
from models.heads.similarity_head import SimilarityHead
from models.recognizer.similarity_recognizer import SimilarityRecognizer3D