# -*- coding: utf-8 -*-
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K
# from tensorflow.keras.activations import sigmoid
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
import warnings
warnings.filterwarnings('ignore')   #忽略警告信息
from keras import layers
from keras import backend as K
from keras.activations import sigmoid
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import numpy as np

#Encoder
#定义一个可以接收任意数量关键词参数的**kwargs
def Conv(filters,kernel_size,stride=(1, 1),d=(1, 1),pd = "same",initializer="he_normal",**kwargs):

    def layer(x):
        x1 = Conv2D(filters, kernel_size, strides=stride,dilation_rate = d, padding = pd, kernel_initializer=initializer)(x)
        x1 = BatchNormalization()(x1,training=False)
        x1 = ReLU()(x1)
        
        out = x1
        return out

    return layer

def side_branch(factor):

    def layers(x):
        x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

        kernel_size = (2*factor, 2*factor)
        x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

        return x
    return layers

def EBGM(factor):

    def layers(inputs):
        A = Conv2D(1, 1, activation = 'sigmoid')(inputs) #224 1
        B = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(A) #same dim 224 valid
        sub = Subtract()([A, B])
        # A_B = subtract([A, B])
        # add1 = add([subtract1,x1])
        kernel_size = (2*factor, 2*factor)
        x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(sub)
        return x

    return layers

def Sub_block():

    def layers(x):

        A = Conv2D(1, 1, activation = 'sigmoid')(x) #(x x 1)
        B = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(A) #same dim 224 valid
        sub = Subtract()([A, B])

        return sub
    return layers

def New_EBGM(factor, filters, mode = "upsampling"):

    #inp, B_skip / deep, shallow
    def layers(inp, B_skip):

        if mode == "transpose":
            inp = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            inp = BatchNormalization(inp)
            inp = ReLU()(inp)
        elif mode == "upsampling":
            inp = UpSampling2D(size=2)(inp) 
#             B_skip = UpSampling2D(size=2)(B_skip) 
        elif mode == "none":
            inp = inp
            B_skip = B_skip
        else:
            raise ValueError()

        Cat_inp_B_skip = Concatenate(axis=3)([inp, B_skip]) #w 28 28
        x0 = Conv2D(filters, 1)(Cat_inp_B_skip)

#         x0 = Double_conv(filters, kernel_size=3)(x0)

        #sub_max
        sub1 = Sub_block()(x0) #(x x 1)
        sub2 = Sub_block()(x0)
        sub3 = Sub_block()(x0)

        Mul_sub1_2 = Multiply()([sub1, sub2]) #x x 1
        SA = Conv2D(1, 1, activation = 'softmax')(Mul_sub1_2)
        Mul_sub1_2_3 = Multiply()([SA, sub3])

        Mul_sub1_2_3_cat = Multiply()([Mul_sub1_2_3, x0])
        Add_out = Add()([Mul_sub1_2_3_cat,x0]) 
        
        x = Conv2D(1, (1, 1), activation=None, padding='same')(Add_out)
        kernel_size = (2*factor, 2*factor)
        Side_Boundary = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
#         print("type_EBAM:",type(Add_out),type(Side_Boundary))
        return Add_out,Side_Boundary

    return layers

def New_RAM(factor,mode = "upsampling"):

    def layers(inp, R_skip):

        if mode == "transpose":
            A = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            A = BatchNormalization(A)
            A = ReLU()(A)
        elif mode == "upsampling":
            A = UpSampling2D(size=2)(inp) 
        else:
            raise ValueError()
        print("A.shape:",A.shape)
        S = Conv2D(1, 1, activation = 'sigmoid')(A) #224 1 data_format: 字符串， channels_last (默认)
        print("1.S.shape:",S.shape) #
        ########
        
#         one_array=np.ones((int(S.shape[1]),int(S.shape[2]),1), dtype=np.float32)
#         one_array = np.expand_dims(one_array,axis=0)
#         one_array = one_array.reshape(S.shape).astype('float32')
#         print("2.one_array:",one_array.shape) #x x 1
        # 将one_array转化成Tensor
#         tensor_one=tf.convert_to_tensor(one_array)
#         tensor_one = Reshape((1,) +(one_array.shape))(tensor_one)
#         tensor_one = np.expand_dims(tensor_one,axis=0)
#         tensor_one = K.ones_like(
#                         S,
#                         dtype=tf.float32
#                         )
#         tensor_one=tf.convert_to_tensor(tensor_one)
        tensor_one = Lambda(K.ones_like)(S)
        print("3.tensor_one:",tensor_one.shape) #
        
        RA = Subtract()([tensor_one, S])
        Mul = Multiply()([R_skip, RA])
        print("R_skip.shape:",R_skip.shape)
        A = Conv2D(int(R_skip.shape[3]), (1, 1), activation=None, padding='same')(A)
        print("Mul.shape:",Mul.shape)
        print("A,shape:",A.shape)
        Add_out = Add()([Mul,A]) 
        print("Add_out.shape:",Add_out.shape)
        
        print("-"*20)
        x = Conv2D(1, (1, 1), activation=None, padding='same')(Add_out)
        kernel_size = (2*factor, 2*factor)
        Side_Region = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
        print("type_RAM:",type(Add_out),type(Side_Region))
        return Add_out,Side_Region

    return layers

def Ms_conv(filters):
    
    def layer(x):
        # x1 = Conv(filters, kernel_size = 1)(x)
        # x3 = Conv(filters, kernel_size = 3)(x)
        # x5 = Conv(filters, kernel_size = 5)(x)
        # ###concat 1*1 3*3 5*5
        # concat1_3_5 = concatenate([x1,x3,x5],axis=3) #filters*3
        x3_d1 = Conv(filters, kernel_size = 3,d=(1, 1))(x)
        x3_d2 = Conv(filters, kernel_size = 3,d=(2, 2))(x)
        x3_d4 = Conv(filters, kernel_size = 3,d=(4, 4))(x)
        #concat 3*3 d=1,d=2,d=4
        concatd1_d2_d4 = concatenate([x3_d1,x3_d2,x3_d4],axis=3) #filters*3

        #1*1
        y = Conv(filters, kernel_size = 1)(concatd1_d2_d4)

        out = y
        return out

    return layer

def Asy_conv(filters, kernel_size):
    
    def layer(x):
        
        cx_11 = Conv(filters, 1)(x)

        c33 = Conv(filters, kernel_size)(cx_11)
        c33_11 = Conv(filters, 1)(c33)
        
        c31 = Conv(filters, (kernel_size,1))(cx_11)
        c13 = Conv(filters, (1,kernel_size))(c31)
        c13_11 = Conv(filters, 1)(c13)

        c13_2 = Conv(filters, (1,kernel_size))(cx_11)
        c31_2 = Conv(filters, (kernel_size,1))(c13_2)
        c31_2_11 = Conv(filters, 1)(c31_2)
        
        concat1 = concatenate([c33_11,c13_11,c31_2_11],axis=3) #filters*3
        concat1_11 = Conv(filters, 1)(concat1)
        
        # short_cut = add([x,concat1_11])

        out = concat1_11
        return out
    
    return layer

def Double_conv(filters, kernel_size):
    
    def layer(x):
        x1 = Conv(filters, kernel_size)(x)
        x2 = Conv(filters, kernel_size)(x1)
        
        out = x2
        return out
    
    return layer

#定义一个可以接收任意数量关键词参数的**kwargs
def DWconv(filters,kernel_size,stride=(1, 1),d=(1, 1),pd = "same",initializer="he_normal",**kwargs):

    def layer(x):
        x1 = DepthwiseConv2D(kernel_size, strides=(1, 1), padding=pd, data_format=None, activation=None, use_bias=False, depthwise_initializer=initializer)(x)
        x1 = BatchNormalization()(x1,training=False)
        x1 = ReLU()(x1)

        out = x1
        return out

    return layer

def Double_DWconv(filters, kernel_size):
    
    def layer(x):
        x1 = DWconv(filters, kernel_size)(x)
        x2 = DWconv(filters, kernel_size)(x1)
        
        out = x2
        return out
    
    return layer

#parallel 并
def parallel(filters, kernel_size, dilation_list=[1,3,5], ratio=16):
    #input c=512
    #filters c=1024
    def layer(input):

        #spatial
        sa_d1 = Conv(filters,kernel_size,d=dilation_list[0])(input)
        sa_d2 = Conv(filters,kernel_size,d=dilation_list[1])(input)
        sa_d3 = Conv(filters,kernel_size,d=dilation_list[2])(input)
        concatd1_d2_d3 = Concatenate(axis=3)([sa_d1,sa_d2,sa_d3])
#         sa_concatd1_d2_d3 = Conv2D(filters,1)(sa_concatd1_d2_d3) #add at 11.03
        weight_d1_d2_d3 = Conv2D(filters,kernel_size=1,activation='softmax',kernel_initializer='he_normal',use_bias=False)(concatd1_d2_d3)
        weight_d1 = Multiply()([weight_d1_d2_d3, sa_d1])
        weight_d2 = Multiply()([weight_d1_d2_d3, sa_d2])
        weight_d3 = Multiply()([weight_d1_d2_d3, sa_d3])
        add_1 = Add()([weight_d1,weight_d2,weight_d3]) #y 01.30 10:08 加入残差input test 3_fold——down/1_fold--down 

        # channel
#         ca_d1 = Conv(filters,kernel_size,d=dilation_list[0])(input)
#         ca_d2 = Conv(filters,kernel_size,d=dilation_list[1])(input)
#         ca_d3 = Conv(filters,kernel_size,d=dilation_list[2])(input)
#         ca_concatd1_d2_d3 = Concatenate(axis=3)([ca_d1,ca_d2,ca_d3])
#         ca_concatd1_d2_d3 = Conv2D(filters,1)(ca_concatd1_d2_d3) #add at 11.03
        #se
        gap = GlobalAveragePooling2D(name='gap')(concatd1_d2_d3)
        fc_1 = Dense(filters//ratio,activation='relu',kernel_initializer='he_normal', use_bias=False)(gap)
        se_weights = Dense(filters,activation='sigmoid')(fc_1)
        # se_weights = Conv2D(filters, kernel_size = 1)(fc_2)
        se_d1 = Multiply()([sa_d1, se_weights])
        se_d2 = Multiply()([sa_d2, se_weights])
        se_d3 = Multiply()([sa_d3, se_weights])
        add_2= Add()([se_d1,se_d2,se_d3]) #y 01.30 10:08 加入残差input test 3_fold——down/1_fold--down 

        cat = Concatenate(axis=3)([add_1,add_2])

        output = cat

        return output

    return layer
    
 # serial 串 未改   
def serial(filters, kernel_size, dilation_list=[1,3,5], ratio=16):
    #input c=512
    #filters c=1024
    def layer(input):

        #spatial
        sa_d1 = Conv(filters,kernel_size,d=dilation_list[0])(input)
        sa_d2 = Conv(filters,kernel_size,d=dilation_list[1])(input)
        sa_d3 = Conv(filters,kernel_size,d=dilation_list[2])(input)
        sa_concatd1_d2_d3 = concatenate([sa_d1,sa_d2,sa_d3],axis=3)
        weight_d1_d2_d3 = Conv2D(filters,kernel_size=1,activation='softmax',kernel_initializer='he_normal',use_bias=False)(sa_concatd1_d2_d3)
        weight_d1 = multiply([weight_d1_d2_d3, sa_d1])
        weight_d2 = multiply([weight_d1_d2_d3, sa_d2])
        weight_d3 = multiply([weight_d1_d2_d3, sa_d3])
        add_1 = add([weight_d1,weight_d2,weight_d3]) #y 01.30 10:08 加入残差input test 3_fold——down/1_fold--down 

        # channel
        ca_d1 = Conv(filters,kernel_size,d=dilation_list[0])(add_1)
        ca_d2 = Conv(filters,kernel_size,d=dilation_list[1])(add_1)
        ca_d3 = Conv(filters,kernel_size,d=dilation_list[2])(add_1)
        ca_concatd1_d2_d3 = concatenate([ca_d1,ca_d2,ca_d3],axis=3)
        #se
        gap = GlobalAveragePooling2D(name='gap')(ca_concatd1_d2_d3)
        fc_1 = Dense(filters//ratio,activation='relu',kernel_initializer='he_normal', use_bias=False)(gap)
        se_weights = Dense(filters,activation='sigmoid')(fc_1)
        # se_weights = Conv2D(filters, kernel_size = 1)(fc_2)
        se_d1 = multiply([ca_d1, se_weights])
        se_d2 = multiply([ca_d2, se_weights])
        se_d3 = multiply([ca_d3, se_weights])
        add_2= add([se_d1,se_d2,se_d3]) #y 01.30 10:08 加入残差input test 3_fold——down/1_fold--down 

        output = add_2

        return output

    return layer

######-----------------------------------------------------------
def encoder(pre):

    def layer(inputs):
        
#         name = inputs
        index = inputs
        #pre_trained_block
#         pre_trained = pre.get_layer(name = name).output
        pre_trained = pre.get_layer(index=index).output
        print(type(pre_trained))
        print(pre_trained)
        return pre_trained
    
    return layer

#decoder
def unet_decoder(filters,mode = "upsampling"):
    
    def layer(inputs):
        inp, skip3 = inputs
        if mode == "transpose":
            x = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            x = BatchNormalization(x)
            x = ReLU()(x)
        elif mode == 'upsampling':
            x = UpSampling2D(size=2)(inp) #w #28
        else:
            raise ValueError()
        
        # skip3 = UpSampling2D(size=2)(skip3) #28
        concat3 = Concatenate(axis=3)([x, skip3]) #w 28 28
        
#         x3 = Conv2D(filters, kernel_size=3,padding = "same",kernel_initializer="he_normal",
#         kernel_regularizer=regularizers.l2(weight_decay))(concat3) #w 原Conv
        x3 = Double_conv(filters, kernel_size=3)(concat3)
        out = x3
        
        return out

    return layer


#res_decoder
def res_decoder(filters,mode = "upsampling"):
    
    def layer(inputs):
        inp, skip3 = inputs
        if mode == "transpose":
            x = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            x = BatchNormalization(x)
            x = ReLU()(x)
        elif mode == 'upsampling':
            x = UpSampling2D(size=2)(inp) #w #28
        else:
            raise ValueError()
        
        concat3 = Concatenate(axis=3)([x, skip3]) #w 28 28 2432
        x3 = Double_conv(filters, kernel_size=3)(concat3)
        short_cut = Conv(filters,1)(concat3) #
        add_1 = Add()([x3,short_cut])
        add_1 = ReLU()(add_1)
        out = add_1
        
        return out

    return layer
