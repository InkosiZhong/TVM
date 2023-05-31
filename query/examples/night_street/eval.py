import os, sys
sys.path.append('.')
from codec import VPFDecoder, CV2Decoder
from tile import str2rect
from query.examples.night_street.gen_embed import NightStreetTiledConfig, NightStreetTiledIndex
from query import AggregateQuery, LimitQuery, TrackQuery, SUPGPrecisionQuery, SUPGRecallQuery
from query.examples.utils import create_score_func
import json

# Feel free to change this!
ROOT_DATA_DIR = '../datasets/jackson-town-square/'
TRAIN_DATA = os.path.join(ROOT_DATA_DIR, '2017-12-14-tile')
TEST_DATA = os.path.join(ROOT_DATA_DIR, '2017-12-17-tile')
CACHE_ROOT = 'query/examples/night_street/cache'
DEC_TYPE = VPFDecoder

tile_config = json.load(open('config/night_street.json', 'r'))
valid_area = str2rect(tile_config['valid_area'])
ban_area = [str2rect(x) for x in tile_config['ban_area']]
score_func = create_score_func(valid_area, ban_area, ['car'])

class NightStreetAggregateQuery(AggregateQuery):
    def score(self, target_dnn_output):
        return score_func(target_dnn_output)
    
class NightStreetLimitQuery(LimitQuery):
    def score(self, target_dnn_output):
        return score_func(target_dnn_output)
    
class NightStreetTrackQuery(TrackQuery):
    def score(self, target_dnn_output):
        return score_func(target_dnn_output)
    
class NightStreetSUPGPrecisionQuery(SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if score_func(target_dnn_output) > 0 else 0.0
    
class NightStreetSUPGRecallQuery(SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if score_func(target_dnn_output) > 0 else 0.0
        
import time
def init_index():
    os.system('sh scripts/clean_cache.sh')
    t = time.time()
    config = NightStreetTiledConfig().eval()
    config.cache_root = CACHE_ROOT
    config.train_data = TRAIN_DATA
    config.test_data = TEST_DATA
    config.dec_type = DEC_TYPE
    index = NightStreetTiledIndex(config)
    index.init()
    print(f'init cost {time.time() - t:.2f}s')
    return index

if __name__ == '__main__':
    # Aggregation
    index = init_index()
    query = NightStreetAggregateQuery(index)
    query.execute(err_tol=0.01, confidence=0.05, batch_size=100)

    # Limit
    index = init_index()
    t = time.time()
    query = NightStreetLimitQuery(index)
    query.execute(want_to_find=1, nb_to_find=1000, GAP=0, batch_size=100)

    # Recall
    index = init_index()
    query = NightStreetSUPGRecallQuery(index)
    query.execute(budget=10000, batch_size=100)
    
    # Tracking
    index = init_index()
    query = NightStreetTrackQuery(index)
    query.execute(workload_path='query/miris/cache/night_street.txt', batch_size=100)