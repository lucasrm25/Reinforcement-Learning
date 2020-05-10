import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensornets as nets

# tensorboard --logdir logdir


model_path = "saved_models_inception/"
model_name = model_path + 'model'
logdir_name = 'logdir_inception'

class Autoencoder(object):

    def __init__(self):
        with tf.name_scope('Input'):
            self.image = tf.placeholder(tf.float32, [None, 400, 600, 3], name='image')
            self.resized_image = tf.image.resize_images(self.image, [400, 400])
        tf.summary.image('resized_image', self.resized_image, 4)
        
        self.inception, self.z = self.encoder(self.resized_image)
        tf.summary.histogram("encoded", self.z)
        
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, 4)
          
        self.loss = self.compute_loss_MSE()           
        tf.summary.scalar('loss_train_MSE', self.loss)
                
        self.merged = tf.summary.merge_all()


    def encoder(self, x):
        print( '{0} Input {1}'.format(0, x.shape.as_list()) )        
        inceptionV3_conv = nets.Inception3(x, stem=True, is_training=False, classes=100)
        print( '{0} Inception {1}'.format(1, inceptionV3_conv.shape.as_list()) )
        with tf.name_scope('Encoder'):
            x = tf.layers.average_pooling2d(inceptionV3_conv, pool_size=11, strides=1, padding='valid')
            print( '{0} AveragePooling {1}'.format(2, x.shape.as_list()) )            
            x = tf.layers.flatten(x)
            z = tf.layers.dense(x, units=1024, name='trainable/z', activation=tf.nn.sigmoid)
            print( '{0} Dense {1}'.format(3, z.shape.as_list()) )
        return inceptionV3_conv, z

    def decoder(self, z):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()   
        with tf.name_scope('Decoder'):
            x = tf.layers.dense(z, 2048, activation=None, name='trainable/z_y')
            print( '{0} Dense {1}'.format(4, x.shape.as_list()) )
            x = tf.reshape(x, [-1, 1, 1, 2048])
            x = tf.layers.conv2d_transpose(x, filters=1024, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc0', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc1', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc2', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(7, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc3', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(8, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc4', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(9, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu, name='trainable/dc5', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(10, x.shape.as_list()) )
            x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid, name='trainable/dc6', kernel_initializer=initializer, bias_initializer=initializer)
            print( '{0} DeConv2d {1}'.format(11, x.shape.as_list()) )
        return x
    
    def compute_loss_MSE(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1) )   
        return reconstruction_loss


#%%

def data_iterator(batch_size):
	data_files = glob.glob('./data/obs_data_VAE_*')
	while True:
		data = np.load(random.sample(data_files, 1)[0])
		np.random.shuffle(data)
		np.random.shuffle(data)
		N = data.shape[0]
		start = np.random.randint(0, N-batch_size)
		yield data[start:start+batch_size]


def train_vae(alpha=1e-3, steps=2, epochs=5, batch_size=64):
    tf.reset_default_graph()
    
    with tf.Session() as sess:

        global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.variables_initializer([global_step]).run()
    
        log_main_path = './'+logdir_name
        if not os.path.isdir(log_main_path):
            os.mkdir(log_main_path)
        dirs = [os.path.join(log_main_path, o) for o in os.listdir(log_main_path) if os.path.isdir(os.path.join(log_main_path,o))]
        run_number = len(dirs)    
        
        writer = tf.summary.FileWriter('{0}/run_{1:02.0f}'.format(logdir_name,run_number), sess.graph)
    
        network = Autoencoder()
        optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"trainable") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"inception3/block7")
        train_op = tf.train.AdamOptimizer(alpha).minimize(network.loss, 
                                                          global_step=global_step,
                                                          var_list=optimizer_scope)
        tf.global_variables_initializer().run()
#        tf.variables_initializer(optimizer_scope).run()
        sess.run(network.inception.pretrained())
    
        saver = tf.train.Saver(max_to_keep=1)
        step = global_step.eval()
        training_data = data_iterator(batch_size=batch_size)
    
        try: # if tf.train.latest_checkpoint(model_path) == None
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print("Model restored from: {}".format(model_path))
        except:
            print("Could not restore saved model")
    
        try:
            for _ in range(steps):
                optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"beta")
                tf.variables_initializer(optimizer_scope).run()
                
                for epoch in range(epochs):
                    images = next(training_data)
                    
#                    if epoch % 100 == 0:
#                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                        run_metadata = tf.RunMetadata()
#                        summary, _ = sess.run([train_op, network.loss, network.merged],
#                                              feed_dict={network.image: images},
#                                              options=run_options,
#                                              run_metadata=run_metadata)
#                        writer.add_run_metadata(run_metadata, 'step%d' % global_step)
#                        print('Adding run metadata for', epoch)
#                    else:
                    _,loss_value,summary = sess.run([train_op, network.loss, network.merged],
                                                    feed_dict={network.image: images})
                        
                    writer.add_summary(summary, step)
                    print ('step {}: training loss {:.6f}'.format(step, loss_value))
        
                    if np.isnan(loss_value):
                        raise ValueError('Loss value is NaN')
                    if step % 10 == 0 and step > 0:                
                        save_path = saver.save(sess, model_name, global_step=global_step)
                    if loss_value <= 35:
                        print ('step {}: training loss {:.6f}'.format(step, loss_value))
                        save_path = saver.save(sess, model_name, global_step=global_step)
                        break
                    step+=1
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interrupt")
        except Exception as e:
            print("Exception: {}".format(e))


if __name__ == '__main__':
#    alpha_v = [1e-4,1e-5,1e-6,1e-7]
#    while True:
#        for alpha in alpha_v:
            alpha=10e-5
            train_vae(alpha=alpha, steps=10, epochs=300, batch_size=50)
#    load_vae()
