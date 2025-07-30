import os

import tensorflow as tf
import numpy as np
import os
import random
from utilities import getAvailableModels
from models.efficientnet import buildModel

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
    selected_model_config = [c for c in configs if c['name'].lower() == name][0]
    path = os.path.join(os.getcwd(), 'models', selected_model_config['weight_filename'])
    model = buildModel()
    print("loading weights from: ", path)
    model.load_weights(path, skip_mismatch=True)
    return model