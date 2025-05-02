from .efficientnet_model_builder import get_pretrained_model as get_pretrained_model_efficientnet
import os

import tensorflow as tf
import numpy as np
import os
import random
from utilities import getAvailableModels

seed = 999

def seedEverything(seed, deterministic = False):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    tf.keras.utils.set_random_seed(seed)
    
    if deterministic:
        tf.config.experimental.enable_op_determinism()
        
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        #tf.config.threading.set_inter_op_parallelism_threads(1)
        #tf.config.threading.set_intra_op_parallelism_threads(1)

## This step is necessary to ensure that the models behave as intended; as they did during training and testing.
seedEverything(seed)

def load_model(name):
    configs = getAvailableModels()
    if name == 'efficientnetv2s':
        config = [c for c in configs if c['name'].lower() == name][0]
        model = get_pretrained_model_efficientnet(os.path.join(os.getenv('dash_app_root'), config['path']))
    return model