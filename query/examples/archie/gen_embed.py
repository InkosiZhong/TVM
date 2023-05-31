import os, sys
import torch
import torchvision
sys.path.append('.')
from codec import VPFDecoder, CV2Decoder
from tile import TileFullScanDataset, TileRandomSelectDataset, str2rect
from tile.utils import LRUBuffer
from index import Index, IndexConfig
from query.detector import YoloV5, MaskRCNN
from query.examples.utils import count_is_close_helper, create_embedding_dnn_transform_fn
import json

# Feel free to change this!
ROOT_DATA_DIR = '../datasets/archie-day/'
TRAIN_DATA = os.path.join(ROOT_DATA_DIR, '2018-04-09-tile')
TEST_DATA = os.path.join(ROOT_DATA_DIR, '2018-04-10-tile')
DEC_TYPE = VPFDecoder

def obj_map_func(obj):
    if obj.object_name == 'truck':
        obj.object_name = 'car'
    return obj

class ArchieTiledIndex(Index):
    def get_target_dnn(self):
        model = YoloV5(model_capacity='yolov5m', map_func=obj_map_func).cuda()
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
    def get_pretrained_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train=True, random_select=False):
        video_fp = self.config.train_data if train else self.config.test_data
        if random_select:
            dataset = TileRandomSelectDataset(
                video_fp,
                self.config.dec_type, 
                target_size=None,
                fast_seek=True,
                codec_cache_size=0,
                frame_cache_size=0,
                cache_update_alg=LRUBuffer,
                cache_unhit=False
            )
        else:
            dataset = TileFullScanDataset(
                video_fp,
                self.config.dec_type, 
                target_size=None,
                max_workers=4,
                I_frame_only=False
            )
        return dataset
    
    def get_embedding_dnn_dataset(self, train=True, random_select=False):
        video_fp = self.config.train_data if train else self.config.test_data
        if random_select:
            dataset = TileRandomSelectDataset(
                video_fp,
                self.config.dec_type, 
                target_size=(224, 224),
                fast_seek=True,
                codec_cache_size=0,
                frame_cache_size=0,
                cache_update_alg=LRUBuffer,
                cache_unhit=False,
                #transform_fn=embedding_dnn_transform_fn
            )
        else:
            dataset = TileFullScanDataset(
                video_fp,
                self.config.dec_type, 
                target_size=(224, 224),
                max_workers=4,
                I_frame_only=False,
                #transform_fn=embedding_dnn_transform_fn
            )
        return dataset
    
    def target_dnn_callback(self, target_dnn_output):
        label = []
        threshold = 0.5
        labels = ['car', 'person']
        for output in target_dnn_output[0]:
            if output.conf > threshold and output.object_name in labels:
                label.append(output)
        return label
        
    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = count_is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True

class ArchieTiledConfig(IndexConfig):
    def __init__(self):
        super().__init__()
        self.cache_root = 'query/examples/archie/cache'
        self.train_data = TRAIN_DATA
        self.test_data = TEST_DATA
        self.dec_type = DEC_TYPE
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 60000

if __name__ == '__main__':
    config = ArchieTiledConfig()
    index = ArchieTiledIndex(config)
    index.init()