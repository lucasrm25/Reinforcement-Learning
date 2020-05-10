import tensorflow as tf
import numpy as np
import tensornets as nets

#%%

nFilters = lambda level: 2**(np.log2(32)+level)

w = 400
h = 400
d = 3

F = 4
S = 2 


#%% 

x= np.reshape(np.random.sample(w*h*d), [w,h,d]).astype('float32')
x = tf.constant(x)
x = tf.reshape(x, [1, w, h, 3])
print( '{0} Input {1}'.format(0, x.shape.as_list()) )

i=0
while min(x.shape.as_list()[1:3]) > 2:    # (W - F + 2*P)/S + 1
    x = tf.layers.conv2d(x, filters=nFilters(i), kernel_size=F, strides=S, padding='valid', activation=tf.nn.relu)
    print( '{0} Conv2d {1}'.format(i+1, x.shape.as_list()) )
#    if min(x.shape.as_list()[1:3]) <= 2: break
#    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#    print( '{0} Pooling {1}'.format(i+1, x.shape.as_list()) )
    i+=1

x = tf.layers.flatten(x)
print( '{0} Flatten {1}'.format(i+1, x.shape.as_list()) )

x = tf.reshape(x, [-1, 1, 1, int(x.shape.as_list()[1]/1) ])
print( '{0} Reshape {1}'.format(i+1, x.shape.as_list()) )


F = [5,6,5,6,6,6,6]# [4,3,3,3,3]
S = [2,2,2,2,2,2,2,2]
#i-=1
j=0
while max(x.shape.as_list()[1:3]) < 400:    # (W - F + 2*P)/S + 1    
    x = tf.layers.conv2d_transpose(x, filters=nFilters(i-1), kernel_size=F[j], strides=S[j], padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(i+1, x.shape.as_list()) )  
#    if max(x.shape.as_list()[1:3]) >= 400: break
#    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#    print( '{0} Upsampling {1}'.format(i+1, x.shape.as_list()) )
    i-=1
    j+=1


#%%
   
    
x= np.reshape(np.random.sample(w*h*d), [w,h,d]).astype('float32')
x = tf.constant(x)
x = tf.reshape(x, [1, w, h, d])
print( '{0} Input {1}'.format(0, x.shape.as_list()) )
   
x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(1, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(2, x.shape.as_list()) )
x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
print( '{0} Pooling {1}'.format(3, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(4, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(5, x.shape.as_list()) )
x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
print( '{0} Pooling {1}'.format(6, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=512, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(7, x.shape.as_list()) )

x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
print( '{0} Upsampling {1}'.format(4, x.shape.as_list()) )
x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} DeConv2d {1}'.format(3, x.shape.as_list()) )
x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} DeConv2d {1}'.format(2, x.shape.as_list()) )
x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
print( '{0} Upsampling {1}'.format(1, x.shape.as_list()) )
x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )

#x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
#print( '{0} DeConv2d {1}'.format(-1, x.shape.as_list()) )



#%% Inception
tf.reset_default_graph()

with tf.Session() as sess:
    
    x= np.reshape(np.random.sample(w*h*d), [w,h,d]).astype('float32')
    x = tf.constant(x)
    x = tf.reshape(x, [1, w, h, d])
    print( '{0} Input {1}'.format(0, x.shape.as_list()) )
    
    
    #inceptionV3_conv = tf.keras.applications.InceptionV3(input_tensor=x, weights='imagenet', include_top=False, pooling='avg', input_shape=(400, 400, 3))   
    #print( '{0} Inception {1}'.format(6, inceptionV3_conv.layers[-1].output_shape ) )
    
    inceptionV3_conv = nets.Inception3(x, stem=True, is_training=False, classes=100)
    #nets.pretrained(inceptionV3_conv)
    print( '{0} Inception {1}'.format(1, inceptionV3_conv.shape.as_list()) )
    
    trainable_vars = tf.trainable_variables()
    print(len(trainable_vars))
    
    x = tf.layers.average_pooling2d(inceptionV3_conv, pool_size=11, strides=1, padding='valid')
    print( '{0} AveragePooling {1}'.format(2, x.shape.as_list()) )
    
    
    x = tf.layers.conv2d_transpose(x, filters=1024, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(3, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(2, x.shape.as_list()) )
    x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
    print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
    
    
#    x = tf.layers.conv2d_transpose(inceptionV3_conv, filters=256, kernel_size=4, strides=1, padding='valid', activation=tf.nn.relu, name='trainable/dc1')
#    print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
#    x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=1, padding='valid', activation=tf.nn.relu, name='trainable/dc2')
#    print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
#    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#    print( '{0} Upsampling {1}'.format(4, x.shape.as_list()) )
#    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, name='trainable/dc3')
#    print( '{0} DeConv2d {1}'.format(3, x.shape.as_list()) )
#    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc4')
#    print( '{0} DeConv2d {1}'.format(2, x.shape.as_list()) )
#    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#    print( '{0} Upsampling {1}'.format(1, x.shape.as_list()) )
#    x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc5')
#    print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
#    x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, name='trainable/dc6')
#    print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
    
    
    optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"trainable")
    
    trainable_vars = tf.trainable_variables()
    print(len(trainable_vars))


#%% 
import tensorflow as tf
import numpy as np
import tensornets as nets
    
inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv2(inputs, nets.Darknet19)
model.print_middles()

w = 416
h = 416
d = 3

x= np.reshape(np.random.sample(w*h*d), [w,h,d]).astype('float32')
x = tf.constant(x, name='const')
x = tf.reshape(x, [1, w, h, d], name='reshape')
print( '{0} Input {1}'.format(0, x.shape.as_list()) )

#x = tf.image.resize_images(x, [416, 416])

yolo = nets.YOLOv2( x=x, stem_fn=nets.Darknet19, stem_out=None, is_training=False)
print( '{0} YOLO {1}'.format(1, yolo.shape.as_list()) )

yolo.print_summary()
   
x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(1, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(2, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(4, x.shape.as_list()) )
x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
print( '{0} Conv2d {1}'.format(5, x.shape.as_list()) )


