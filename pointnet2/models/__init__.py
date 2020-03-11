from pointnet2.models.pointnet2_msg_cls import PointNet2ClassificationMSG
from pointnet2.models.pointnet2_msg_sem import PointNet2SemSegMSG
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG

models = {
    "ssg-classification": PointNet2ClassificationSSG,
    "msg-classification": PointNet2ClassificationMSG,
    "ssg-semantic_segmentation": PointNet2SemSegSSG,
    "msg-semantic_segmentation": PointNet2SemSegMSG,
}
