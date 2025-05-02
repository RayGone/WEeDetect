import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, Reshape, \
Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Average
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.applications import efficientnet_v2

def ChannelMaxPooling(x, pool_size='infer',strides=None, layer_num=1):
    """
      pool_size If not "infer", then the final output size will be {h*w*(c/pool_size)}., where "c" is feature size.
      "infer" works only if {c%(h*w) == 0}; if true, then the final output size is equal to "c"

    """
    
    if pool_size == 'infer':
        c = x.shape[-1]
        d = x.shape[1]
        if(d%2 == 1): d+=1
        pool_size = c//d**2
        
    if strides is None:
        strides = pool_size

    _channel_max = layers.Reshape((-1,x.shape[-1]), name="ForwardReshape-CMP-L{}".format(layer_num))(x)
    _channel_max = layers.MaxPool1D(pool_size,strides,padding='same', name="ChannelMax-CMP-L{}".format(layer_num), data_format='channels_first')(_channel_max) 
    _channel_max = layers.Reshape((x.shape[1],x.shape[2],-1), name= "BackwardReshape-CMP-L{}".format(layer_num))(_channel_max)
    
    return _channel_max

def ChannelAvgPooling(x, pool_size='infer',strides=None, layer_num=1):
    """
      pool_size If not "infer", then the final output size will be {h*w*(c/pool_size)}., where "c" is feature size.
      "infer" works only if {c%(h*w) == 0}; if true, then the final output size is equal to "c"

    """
    if pool_size == 'infer':
        c = x.shape[-1]
        d = x.shape[1]
        if(d%2 == 1): d+=1
        pool_size = c//d**2
        
    if strides is None:
        strides = pool_size

    _channel_avg = layers.Reshape((-1,x.shape[-1]), name="ForwardReshape-CAP-L{}".format(layer_num))(x)
    _channel_avg = layers.AveragePooling1D(pool_size,strides,padding='same', name="ChannelAvg-CAP-L{}".format(layer_num), data_format='channels_first')(_channel_avg) 
    _channel_avg = layers.Reshape((x.shape[1],x.shape[2],-1), name= "BackwardReshape-CAP-L{}".format(layer_num))(_channel_avg)
    
    return _channel_avg


"""
 *** Here Spatial Pooling and Channel Pooling are same***
"""
class SpatialMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=2,stride=None,padding='valid',data_format='channels_last', **kwargs):
        super(SpatialMaxPooling2D,self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        if stride is None:
            self.stride = self.pool_size

        self.data_format = 'channels_last' if data_format == 'channels_first' else 'channels_first'
        self.max = tf.keras.layers.MaxPool1D(self.pool_size, self.stride, padding=self.padding, data_format=self.data_format)

    def build(self,input_shape):
        self.reshape_forward = tf.keras.layers.Reshape((-1,input_shape[-1]))
        self.reshape_backward= tf.keras.layers.Reshape((input_shape[1], input_shape[2], -1))

    def call(self,x):
        x = self.reshape_forward(x)
        x = self.max(x)
        x = self.reshape_backward(x)
        return x

class SpatialAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=2,stride=None,padding='valid',data_format='channels_last', **kwargs):
        super(SpatialAveragePooling2D,self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        if stride is None:
            self.stride = self.pool_size

        self.data_format = 'channels_last' if data_format == 'channels_first' else 'channels_first'
        self.avg = tf.keras.layers.AveragePooling1D(self.pool_size, self.stride, padding=self.padding, data_format=self.data_format)

    def build(self,input_shape):
        self.reshape_forward = tf.keras.layers.Reshape((-1,input_shape[-1]))
        self.reshape_backward= tf.keras.layers.Reshape((input_shape[1], input_shape[2], -1))

    def call(self,x):
        x = self.reshape_forward(x)
        x = self.avg(x)
        x = self.reshape_backward(x)
        return x

def AverageOfMaximums(x, max_pool_size=2, layer_num=1):
    _max = layers.MaxPooling2D(pool_size=max_pool_size,padding='same', name="Maximums_L{}".format(layer_num))(x)
    _avg = layers.GlobalAveragePooling2D(name="Average_of_Maximums_L{}".format(layer_num))(_max)
    return _avg

def cbam_block(cbam_feature, ratio=8, layer_num=0):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio, layer_num)
    cbam_feature = spatial_attention(cbam_feature, layer_num)
    return cbam_feature

def channel_attention(input_feature, ratio=8, layer_num=0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                           name='Dense_Squeeze_CA_L{}'.format(layer_num))
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                           name='Dense_Excite_CA_L{}'.format(layer_num))

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 5

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1

def MultiFilterSpatialAttention(input_feature, kernel_size=7, num_pooled_channel=4, layer_num = 0):
    in_shape = input_feature.shape

    pool_size =  in_shape[-1] // num_pooled_channel

    channel_avg = SpatialAveragePooling2D(pool_size, padding='same', name='MFSA_SAP_L{}'.format(layer_num))(input_feature)
    channel_max = SpatialMaxPooling2D(pool_size, padding='same' , name='MFSA_SMP_L{}'.format(layer_num))(input_feature)

    concat = layers.Concatenate(axis=3, name='MFSA_ConcatChannels_L{}'.format(layer_num))([channel_avg, channel_max])

    sa_feature_x = layers.Conv2D(filters = in_shape[-1],
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    groups= num_pooled_channel,
                    use_bias=False,
                    name="MFSA_Conv_L{}".format(layer_num))(concat)

    # sa_shape = sa_feature_x.shape

    sa_feature_x =  layers.multiply([input_feature, sa_feature_x])

    # sa_feature_x = Reshape((in_shape[1], in_shape[2], -1), name="MFSA_OutBackwardReshape_L{}".format(layer_num))(sa_feature_x)
    return sa_feature_x
    return cbam_feature

def build_model(input_shape=(256, 256, 3)):
    model = efficientnet_v2.EfficientNetV2S(input_shape=input_shape, include_top=False, pooling=None, weights=None)
    model.trainable = False
    
    
    POOLING = 'avgmax' # "avg"=Global Average, "max"=Global Max, "avgmax" = average of maximums

    x_in = layers.Input((256,256,3), name='InLayer')
    x = model(x_in)
            

    out_res = AverageOfMaximums(x, layer_num='ResOutPool1')
    ##############################################333
    ######## Squeeze and Excitation Network ######333
    #############################################3333
    x_cbam = MultiFilterSpatialAttention(x, num_pooled_channel=32, layer_num='final')
    x_cbam = channel_attention(x_cbam, layer_num='final')

    if(POOLING == 'avgmax'):
        print("""### Average Of Maximums Scheme for Pooling #333""" )
        x = AverageOfMaximums(x_cbam, layer_num='OutPool')
    elif(POOLING == 'max'):
        print("Using GLobal Max Pooling")
        x = layers.GlobalMaxPool2D(name="Global_Maximum_OutPool")(x_cbam)

    else: #POOLING == 'avg'
        print("Using GLobal Average Pooling")
        x = GlobalAveragePooling2D(name="Global_Average_OutPool")(x_cbam)
    
    x = Add()([x, out_res])
    x = layers.Dense(x.shape[-1] // 2, activation='gelu', name='pre-classifier')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.3, name='dropout0.3')(x) 
    x_out = layers.Dense(9, activation='softmax')(x)

    model = keras.Model(inputs=x_in, outputs=x_out, name='DeepWeeds-EfficientNet-CBAM')
    # model.summary()
    return model

def get_pretrained_model(path=None):
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),'DeepWeeds-EfficientNet-CBAM.keras')
    path = path if path else default
    print("Loading default model")
    model = build_model()
    print("loading weights from: ", path)
    model.load_weights(path, skip_mismatch=True)
    return model