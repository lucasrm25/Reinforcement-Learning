import tensorflow as tf
import numpy as np
import glob, random, os
import tensornets as nets
import sys
# tensorboard --logdir logdir


class CNN_Lander(object):

    def __init__(self):
     
        with tf.name_scope('Input_img'):
            self.input_image = tf.placeholder(tf.float32, [None, 200, 200, 3], name='input_image')
            
        with tf.name_scope('Output_state_true'):
            self.y_true = tf.placeholder(tf.float32, [None, 3], name='y_true')
            
        self.y_pred = self.encoder_YOLO(self.input_image)
             
        with tf.name_scope('Plot_Bounding_Boxes'):
            self.input_image_bounded = self.img_bounding_box(self.input_image, self.y_true)
            tf.summary.image('input_image', self.input_image_bounded, 4)
            self.output_image_bounded = self.img_bounding_box(self.input_image, self.y_pred)
            tf.summary.image('output_image', self.output_image_bounded, 4)
        
        with tf.name_scope('Losses'): 
            self.loss = self.compute_loss_MSE(self.y_true, self.y_pred)           
            tf.summary.scalar('loss_train_MSE', self.loss)
            tf.summary.scalar('loss_train_MAE_x', tf.reduce_mean( tf.abs( self.y_true[:,0] - self.y_pred[:,0] ) ) )
            tf.summary.scalar('loss_train_MAE_y', tf.reduce_mean( tf.abs( self.y_true[:,1] - self.y_pred[:,1] ) ) )
            tf.summary.scalar('loss_train_MAE_teta', tf.reduce_mean( tf.abs( self.y_true[:,2] - self.y_pred[:,2] ) ) )
               
        with tf.name_scope('Optimizers'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.train_op_Adam      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
            self.train_op_Adadelta  = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
            self.train_op_GD        = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
                        
        self.merged = tf.summary.merge_all()

    def img_bounding_box(self, image, y):
        with tf.name_scope('Calculate_Bounding_Boxes'):
            x_pos = tf.add(y[:,0], 1)     # -1=0, 1=2
            x_pos = tf.multiply(x_pos, 0.5)         # 0=0, 2=1
            x_pos = tf.reshape(x_pos, [-1, 1])
            x_pos_min = tf.add(x_pos, -0.05)
            x_pos_max = tf.add(x_pos, 0.05)
            
            y_pos = tf.multiply(y[:,1], -1)    # 0=0, 1=-1 
            y_pos = tf.add(y_pos, 1)                # 0=1, -1=0
            y_pos = tf.multiply(y_pos, 3/4)         # 1=3/4, 0=0
            y_pos = tf.reshape(y_pos, [-1, 1])        
            y_pos_min = tf.add(y_pos, -0.1)
            y_pos_max = tf.add(y_pos, 0)
            
            box_pos = tf.concat([y_pos_min, x_pos_min, y_pos_max, x_pos_max], axis=1)
            box_pos = tf.expand_dims(box_pos, dim=1)        
            output_image = tf.image.draw_bounding_boxes(image, box_pos)
        return output_image


    def encoder_YOLO(self, x):
        print('\nBuilding Graph:')
        initializer_dense = tf.contrib.layers.xavier_initializer()
        print( '\t{0} Input {1}'.format(0, x.shape.as_list()) )
        x = tf.image.resize_images(x, [416, 416])
        self.yolo = nets.TinyYOLOv2(x, nets.TinyDarknet19)
        print( '\t{0} YOLO {1}'.format(1, self.yolo.shape.as_list()) )
        with tf.name_scope('Flatten'):
            x = tf.layers.flatten(self.yolo)
        with tf.name_scope('dense01'):
            x = tf.layers.dense(x, units=512, activation=tf.nn.elu, kernel_initializer=initializer_dense, bias_initializer=initializer_dense)
            print( '\t{0} Dense {1}'.format(8, x.shape.as_list()) )
        with tf.name_scope('dense02'):
            x = tf.layers.dense(x, units=3, activation=None, kernel_initializer=initializer_dense, bias_initializer=initializer_dense)
            print( '\t{0} Dense {1}'.format(10, x.shape.as_list()) )
        return x

    def encoder(self, x):
#        initializer = tf.contrib.layers.xavier_initializer_conv2d() 
        initializer_dense = tf.contrib.layers.xavier_initializer()
        print( '{0} Input {1}'.format(0, x.shape.as_list()) )
        with tf.name_scope('conv2d'):
            x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
            print( '{0} Conv2d {1}'.format(1, x.shape.as_list()) )
            x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
            print( '{0} Conv2d {1}'.format(2, x.shape.as_list()) )
            x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
            print( '{0} Conv2d {1}'.format(4, x.shape.as_list()) )
            x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
            print( '{0} Conv2d {1}'.format(5, x.shape.as_list()) )    
        with tf.name_scope('Flatten'):
            x = tf.layers.flatten(x)
        with tf.name_scope('dense01'):
            x = tf.layers.dense(x, units=512, activation=tf.nn.elu, kernel_initializer=initializer_dense, bias_initializer=initializer_dense)
            print( '{0} Dense {1}'.format(8, x.shape.as_list()) )
        with tf.name_scope('batch_normalization01'):
#            tf.layers.dropout(x,rate=0.5)
            x = tf.layers.batch_normalization(x)
        with tf.name_scope('dense02'):
            x = tf.layers.dense(x, units=128, activation=tf.nn.elu, kernel_initializer=initializer_dense, bias_initializer=initializer_dense)
            print( '{0} Dense {1}'.format(8, x.shape.as_list()) )
        with tf.name_scope('dense03'):
            x = tf.layers.dense(x, units=3, activation=None, kernel_initializer=initializer_dense, bias_initializer=initializer_dense)
            print( '{0} Dense {1}'.format(10, x.shape.as_list()) )
        return x
    
    def compute_loss_MSE(self, y_true, y_pred):
        with tf.name_scope('loss_MSE'):
            weights = np.array([[300,300,100]])
            tf_weights = tf.constant(weights, shape=[1,3], dtype='float32')
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(y_true - y_pred), tf_weights), axis=1) )   
        return reconstruction_loss

#
#a = CNN_Lander()

#%%

def data_iterator(batch_size):
	data_files = glob.glob('./data_CNN_Vision/obs_data_VAE_*')
	while True:
		data = np.load(random.sample(data_files, 1)[0])
		np.random.shuffle(data)
		np.random.shuffle(data)
		N = data.shape[0]
		start = np.random.randint(0, N-batch_size)
		yield data[start:start+batch_size]

def create_log_folder(logdir_name):
    log_main_path = './'+logdir_name
    if not os.path.isdir(log_main_path):
        os.mkdir(log_main_path)
    dirs = [os.path.join(log_main_path, o) for o in os.listdir(log_main_path) if os.path.isdir(os.path.join(log_main_path,o))]
    run_number = len(dirs)
    return run_number


def train_vae(alpha=1e-3, steps=2, epochs=5, batch_size=64):
    
    tf.reset_default_graph()   
    network = CNN_Lander()
    
    run_number = create_log_folder(logdir_name)
    training_data = data_iterator(batch_size=batch_size)
    
    with tf.Session() as sess:
      
        writer = tf.summary.FileWriter('{0}/run_{1:02.0f}'.format(logdir_name,run_number), sess.graph)  # writer.add_graph(sess.graph)
        saver  = tf.train.Saver(max_to_keep=1)

        last_checkpoint = tf.train.latest_checkpoint(model_path)
        if last_checkpoint == None:
            tf.global_variables_initializer().run()
            print("Initializing new model")
            sess.run(network.yolo.pretrained())    # Transfer Learning
            print("Pretrained Yolo loaded")
        else:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print("Model restored from: {0} - global step: {1}".format(model_path, network.global_step.eval()))
            except Exception as e:
                print("\n\nException: {}".format(e))
                sys.exit("Could not restore saved model, graph has changed - choose another model name")
        try:
            step = network.global_step.eval()                 
            for st in range(steps):
#                alpha = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7][st % 13]
                alpha = [1e-6, 1e-6, 1e-6, 1e-6][st % 4]
                
                for epoch in range(epochs):
                    tr_batch = next(training_data)                    
                    tr_x = np.array([x for x in tr_batch[:,0]])
                    tr_y = np.array([x[[0,1,4]] for x in tr_batch[:,1]])
                    
                    _,loss_value,summary = sess.run([network.train_op_Adam, network.loss, network.merged],
                                                    feed_dict={network.input_image: tr_x, network.y_true: tr_y, network.learning_rate: alpha})
                        
                    writer.add_summary(summary, step)
                    print ('step {}: training loss: {:.6f}   alpha:{}'.format(step, loss_value, alpha))
        
                    if np.isnan(loss_value):
                        raise ValueError('Loss value is NaN')
                    if step % 20 == 0 and step > 0:
                        saver.save(sess, model_name, global_step=network.global_step)
                    step+=1
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interrupt")
        except Exception as e:
            print("Exception: {}".format(e))


if __name__ == '__main__':
    exp_name = '_2'
    
    model_path = 'saved_models'+exp_name+'/'
    model_name = model_path + 'model'
    logdir_name = 'logdir' + exp_name
    train_vae(alpha=1e-3, steps=5, epochs=1000, batch_size=50)


