import warnings
warnings.filterwarnings('ignore')   #忽略警告信息
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import efficientnet.keras as efn

#self_define
from losses import *
from layers2 import *

# vgg16 = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3']
# vgg19 = ['block1_conv2','block2_conv2','block3_conv4','block4_conv4','block5_conv4']
vgg16 = [2,5,9,13,17]
vgg19 = [2,5,10,15,20]
res50 = [4,38,80,142,174]
efb7 = [152,255,403,744,802]

filters_list = [64,128,256,512,1024]
IMAGE_SIZE,c = 224,3
input_size=(IMAGE_SIZE,IMAGE_SIZE,c)
VGG16 = VGG16(input_shape = input_size,weights='imagenet',include_top=False)
VGG19 = VGG19(input_shape = input_size,weights='imagenet',include_top=False)
Res50 = ResNet50(input_shape = input_size,weights='imagenet',include_top=False)
Efb7 = efn.EfficientNetB7(input_shape = input_size,weights='imagenet',include_top=False)

# Baseline
# outputs = [outputs]
def Baseline(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((SAM,pre_trained_b4)) #in7 out14 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in14 out28 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in28 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in56 out112 64
    d_x10 = UpSampling2D(size=2)(d_x9) #in112 out224 64
    
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)
    model = Model(inputs = [Pre.input], outputs = [outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_SAM
# outputs = [mid_mask1,outputs]
def B_SAM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    SAM = parallel(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #7 1024
    SAM = Conv(filters_list[4], 1)(SAM) #224 1
    mid_mask1 = side_branch(32)(SAM)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((SAM,pre_trained_b4)) #in7 out14 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in14 out28 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in28 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in56 out112 64
    d_x10 = UpSampling2D(size=2)(d_x9) #in112 out224 64
    
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)
    model = Model(inputs = [Pre.input], outputs = [mid_mask1,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_BGM
# outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,outputs]
def B_BGM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    mid = Conv(filters_list[4], 1)(pre_trained_b5)
    
    #BGM
    add_5,bd5 = New_EBGM(32,filters_list[4],mode = "none")(mid,pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    add_4,bd4 = New_EBGM(16,filters_list[3])(add_5,pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1
    add_3,bd3 = New_EBGM(8,filters_list[2])(add_4,pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1
    add_2,bd2 = New_EBGM(4,filters_list[1])(add_3,pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1
    add_1,bd1 = New_EBGM(2,filters_list[0])(add_2,pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1
    BG_out = UpSampling2D(size=2)(add_1) 
    BG_out = Conv2D(1, 1, activation = 'sigmoid',name='BG_out')(BG_out) #224 1

    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((16,16))(bd_fuse45) #14 1

    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((8,8))(bd_fuse345) #28 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((4,4))(bd_fuse2345) #56 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1
    bd_fuse12345_maxpooling = MaxPool2D((2,2))(bd_fuse12345) #112 1

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in7 out14 512
    d_x6_multiply = Multiply()([d_x6,bd_fuse45_maxpooling]) #14 * 14    #strong_d_x6
    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in14 out28 256
    d_x7_multiply = Multiply()([d_x7,bd_fuse345_maxpooling]) #28 * 28    #strong_d_x7
    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in28 out56 128
    d_x8_multiply = Multiply()([d_x8,bd_fuse2345_maxpooling]) #56 * 56    #strong_d_x8
    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in56 out112 64
    d_x9_multiply = Multiply()([d_x9,bd_fuse12345_maxpooling]) #112 * 112    #strong_d_x9
    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)

    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_RAM
# outputs = [d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs]
def B_RAM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    mid = Conv(filters_list[4], 1)(pre_trained_b5) #224 1

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in7 out14 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in14 out28 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in28 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in56 out112 64

    #mask_side1
    add_6,d_x6_mask2 = New_RAM(16)(mid,d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)
    #mask_side2
    add_7,d_x7_mask3 = New_RAM(8)(add_6,d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)
    #mask_side3
    add_8,d_x8_mask4 = New_RAM(4)(add_7,d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)
    #mask_side4
    add_9,d_x9_mask5 = New_RAM(2)(add_8,d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)
    
    RA_out = UpSampling2D(size=2)(add_9) 
    RA_out = Conv2D(1, 1, activation = 'sigmoid',name='RA_out')(RA_out) #224 1

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)

    model = Model(inputs = [Pre.input], outputs = [d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_SAM_BGM
# outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,outputs]
def B_SAM_BGM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640
    SAM = parallel(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #7 1024
    SAM = Conv(filters_list[4], 1)(SAM) #C==1024 #C/1024 【插值->224*224*1024->numpy.mean->224*224*1->sigmoid->out224*224*1  / numpy.mean->插值->224*224*1】
    mid_mask1 = side_branch(32)(SAM) #C==1
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #BGM
    add_5,bd5 = New_EBGM(32,filters_list[4],mode = "none")(SAM,pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1

    add_4,bd4 = New_EBGM(16,filters_list[3])(add_5,pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    add_3,bd3 = New_EBGM(8,filters_list[2])(add_4,pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    add_2,bd2 = New_EBGM(4,filters_list[1])(add_3,pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    add_1,bd1 = New_EBGM(2,filters_list[0])(add_2,pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    BG_out = UpSampling2D(size=2)(add_1) 
    BG_out = Conv2D(1, 1, activation = 'sigmoid',name='BG_out')(BG_out) #224 1

    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((16,16))(bd_fuse45) #14 1

    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((8,8))(bd_fuse345) #28 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((4,4))(bd_fuse2345) #56 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1
    bd_fuse12345_maxpooling = MaxPool2D((2,2))(bd_fuse12345) #112 1

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((SAM,pre_trained_b4)) #in7 out14 512
    d_x6_multiply = Multiply()([d_x6,bd_fuse45_maxpooling]) #14 * 14    #strong_d_x6
    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in14 out28 256
    d_x7_multiply = Multiply()([d_x7,bd_fuse345_maxpooling]) #28 * 28    #strong_d_x7
    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in28 out56 128
    d_x8_multiply = Multiply()([d_x8,bd_fuse2345_maxpooling]) #56 * 56    #strong_d_x8
    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in56 out112 64
    d_x9_multiply = Multiply()([d_x9,bd_fuse12345_maxpooling]) #112 * 112    #strong_d_x9
    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)

    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_SAM_RAM
# outputs = [mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs]
def B_SAM_RAM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    SAM = parallel(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #7 1024
    SAM = Conv(filters_list[4], 1)(SAM) #224 1   ###### 多通道 1
    mid_mask1 = side_branch(32)(SAM)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1) #输出

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((SAM,pre_trained_b4)) #in7 out14 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in14 out28 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in28 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in56 out112 64

    #mask_side1
    add_6,d_x6_mask2 = New_RAM(16)(SAM,d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2) #输出
    #mask_side2
    add_7,d_x7_mask3 = New_RAM(8)(add_6,d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3) #输出
    #mask_side3
    add_8,d_x8_mask4 = New_RAM(4)(add_7,d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4) #输出
    #mask_side4
    add_9,d_x9_mask5 = New_RAM(2)(add_8,d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5) #输出
    
    RA_out = UpSampling2D(size=2)(add_9) 
    RA_out = Conv2D(1, 1, activation = 'sigmoid',name='RA_out')(RA_out) #224 1

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5]) #C==5 (0,1)
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse) ###### 2 #输出

    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)

    model = Model(inputs = [Pre.input], outputs = [mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# B_BGM_RAM
# outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,
# d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs]
def B_BGM_RAM(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    mid = Conv(filters_list[4], 1)(pre_trained_b5) #224 1
    
    #BGM
    add_5,bd5 = New_EBGM(32,filters_list[4],mode = "none")(mid,pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    add_4,bd4 = New_EBGM(16,filters_list[3])(add_5,pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1
    add_3,bd3 = New_EBGM(8,filters_list[2])(add_4,pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1
    add_2,bd2 = New_EBGM(4,filters_list[1])(add_3,pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1
    add_1,bd1 = New_EBGM(2,filters_list[0])(add_2,pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1
    BG_out = UpSampling2D(size=2)(add_1) 
    BG_out = Conv2D(1, 1, activation = 'sigmoid',name='BG_out')(BG_out) #224 1

    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((16,16))(bd_fuse45) #14 1

    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((8,8))(bd_fuse345) #28 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((4,4))(bd_fuse2345) #56 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1
    bd_fuse12345_maxpooling = MaxPool2D((2,2))(bd_fuse12345) #112 1

    mid_mask1 = side_branch(32)(SAM)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in7 out14 512
    d_x6_multiply = Multiply()([d_x6,bd_fuse45_maxpooling]) #14 * 14    #strong_d_x6
    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in14 out28 256
    d_x7_multiply = Multiply()([d_x7,bd_fuse345_maxpooling]) #28 * 28    #strong_d_x7
    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in28 out56 128
    d_x8_multiply = Multiply()([d_x8,bd_fuse2345_maxpooling]) #56 * 56    #strong_d_x8
    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in56 out112 64
    d_x9_multiply = Multiply()([d_x9,bd_fuse12345_maxpooling]) #112 * 112    #strong_d_x9

    #mask_side1
    add_6,d_x6_mask2 = New_RAM(16)(SAM,d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)
    #mask_side2
    add_7,d_x7_mask3 = New_RAM(8)(add_6,d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)
    #mask_side3
    add_8,d_x8_mask4 = New_RAM(4)(add_7,d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)
    #mask_side4
    add_9,d_x9_mask5 = New_RAM(2)(add_8,d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)
    
    RA_out = UpSampling2D(size=2)(add_9) 
    RA_out = Conv2D(1, 1, activation = 'sigmoid',name='RA_out')(RA_out) #224 1

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)

    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# EBGRANet
# outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,
# mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs]
def EBGRANet(num_class=1,Pre =Efb7,Pre_list=efb7,pretrained_weights=None):

    #Backbone_Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #56 48 
    pre_trained_b1 = UpSampling2D(size=2)(pre_trained_b1) #in56 out112 48
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #28 80
    pre_trained_b2 = UpSampling2D(size=2)(pre_trained_b2) #in28 out56 80
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #14 160
    pre_trained_b3 = UpSampling2D(size=2)(pre_trained_b3) #in14 out28 160
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #7 384
    pre_trained_b4 = UpSampling2D(size=2)(pre_trained_b4) #in7 out14 384
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #7 640

    SAM = parallel(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #7 1024
    SAM = Conv(filters_list[4], 1)(SAM) #224 1
    mid_mask1 = side_branch(32)(SAM)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)
    
    #BGM
    add_5,bd5 = New_EBGM(32,filters_list[4],mode = "none")(SAM,pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    add_4,bd4 = New_EBGM(16,filters_list[3])(add_5,pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1
    add_3,bd3 = New_EBGM(8,filters_list[2])(add_4,pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1
    add_2,bd2 = New_EBGM(4,filters_list[1])(add_3,pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1
    add_1,bd1 = New_EBGM(2,filters_list[0])(add_2,pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1
    BG_out = UpSampling2D(size=2)(add_1) 
    BG_out = Conv2D(1, 1, activation = 'sigmoid',name='BG_out')(BG_out) #224 1

    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((16,16))(bd_fuse45) #14 1

    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((8,8))(bd_fuse345) #28 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((4,4))(bd_fuse2345) #56 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1
    bd_fuse12345_maxpooling = MaxPool2D((2,2))(bd_fuse12345) #112 1

    #res_decoder
    d_x6 = res_decoder(filters_list[3])((SAM,pre_trained_b4)) #in7 out14 512
    d_x6_multiply = Multiply()([d_x6,bd_fuse45_maxpooling]) #14 * 14    #strong_d_x6
    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in14 out28 256
    d_x7_multiply = Multiply()([d_x7,bd_fuse345_maxpooling]) #28 * 28    #strong_d_x7
    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in28 out56 128
    d_x8_multiply = Multiply()([d_x8,bd_fuse2345_maxpooling]) #56 * 56    #strong_d_x8
    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in56 out112 64
    d_x9_multiply = Multiply()([d_x9,bd_fuse12345_maxpooling]) #112 * 112    #strong_d_x9

    #mask_side1
    add_6,d_x6_mask2 = New_RAM(16)(SAM,d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)
    #mask_side2
    add_7,d_x7_mask3 = New_RAM(8)(add_6,d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)
    #mask_side3
    add_8,d_x8_mask4 = New_RAM(4)(add_7,d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)
    #mask_side4
    add_9,d_x9_mask5 = New_RAM(2)(add_8,d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)
    
    RA_out = UpSampling2D(size=2)(add_9) 
    RA_out = Conv2D(1, 1, activation = 'sigmoid',name='RA_out')(RA_out) #224 1

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    d_x10 = UpSampling2D(size=2)(d_x9_multiply) #in112 out224 64
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x10)



    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,BG_out,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,
        mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,RA_out,mask_fuse,outputs])
    
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
