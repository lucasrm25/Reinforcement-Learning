import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensornets as nets

# tensorboard --logdir logdir


model_path = "saved_models_2conv_2/"
model_name = model_path + 'model'
logdir_name = 'logdir_2conv_2'

class Autoencoder(object):

    def __init__(self, VAE=False):
        self.image = tf.placeholder(tf.float32, [None, 400, 600, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [400, 400])
        tf.summary.image('resized_image', self.resized_image, 4)
        
        self.z_x = self.encoder(self.resized_image)
        if VAE:
            self.z, self.z_mu, self.z_logvar = self.encoded_VAE(self.z_x)
        else:
            self.z = self.encoded(self.z_x)
        
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, 4)
          
        if VAE:
            self.loss, self.reconst_loss, self.kl_loss = self.compute_loss_MSE_VAE()
            tf.summary.scalar('loss_train_reconstr', self.reconst_loss)
            tf.summary.scalar('loss_train_kl', self.kl_loss)
        else:
            self.loss = self.compute_loss_MSE()           
        tf.summary.scalar('loss_train_MSE', self.loss)
                
        self.merged = tf.summary.merge_all()


    # Output size is (N-F)/stride +1
#    def encoder(self, x):
#        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(0, x.shape.as_list()) )        
#        x = tf.layers.flatten(x)
#        return x
#    def encoder(self, x):
#        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(1, x.shape.as_list()) )
#        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(2, x.shape.as_list()) )
#        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#        print( '{0} Pooling {1}'.format(3, x.shape.as_list()) )
#        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(4, x.shape.as_list()) )
#        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(5, x.shape.as_list()) )
#        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,padding='valid')
#        print( '{0} Pooling {1}'.format(6, x.shape.as_list()) )
#        x = tf.layers.conv2d(x, filters=512, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} Conv2d {1}'.format(7, x.shape.as_list()) )       
#        x = tf.layers.flatten(x)
#        return x
    
    def encoder(self, x):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()

        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(1, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(2, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(4, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(5, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(7, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=1024, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(7, x.shape.as_list()) )
        x = tf.layers.conv2d(x, filters=2048, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} Conv2d {1}'.format(7, x.shape.as_list()) )
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        return x
    
    def encoder_Inception(self, x):
        print( '{0} Input {1}'.format(0, x.shape.as_list()) )
        inceptionV3_conv = nets.Inception3(x, stem=True, is_training=False)
#        nets.pretrained(inceptionV3_conv)
        print( '{0} Inception {1}'.format(1, inceptionV3_conv.shape.as_list()) )
        x = tf.layers.average_pooling2d(inceptionV3_conv, pool_size=11, strides=1, padding='valid')
        print( '{0} AveragePooling {1}'.format(2, x.shape.as_list()) )
        return x
    
    
    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps
        
    def encoded_VAE(self, x):
        z_mu     = tf.layers.dense(x, units=64, name='z_mu')
        z_logvar = tf.layers.dense(x, units=64, name='z_logvar')
        z        = self.sample_z(z_mu, z_logvar)
        return z, z_mu, z_logvar
    
    def encoded(self, x):
        z = tf.layers.dense(x, units=128, name='z')
        return z

#    def decoder(self, z):
#        x = tf.layers.dense(z, 256, activation=None)
#        x = tf.reshape(x, [-1, 1, 1, 256])
#        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
#        x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
#        print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
#        return x
#    def decoder(self, z):
#        x = tf.layers.dense(z, 512, activation=None)
#        x = tf.reshape(x, [-1, 1, 1, 512])
#        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
#        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
#        x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#        print( '{0} Upsampling {1}'.format(4, x.shape.as_list()) )
#        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=6, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} DeConv2d {1}'.format(3, x.shape.as_list()) )
#        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
#        print( '{0} DeConv2d {1}'.format(2, x.shape.as_list()) )
#        x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#        print( '{0} Upsampling {1}'.format(1, x.shape.as_list()) )
#        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.sigmoid)
#        print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
#        return x
    
    def decoder(self, z):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        
        x = tf.layers.dense(z, 512, activation=tf.nn.sigmoid)
        x = tf.reshape(x, [-1, 1, 1, 512])
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(6, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(5, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(3, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=6, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.relu)
        print( '{0} DeConv2d {1}'.format(2, x.shape.as_list()) )
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', kernel_initializer=initializer, activation=tf.nn.sigmoid)
        print( '{0} DeConv2d {1}'.format(0, x.shape.as_list()) )
        return x
    
    def compute_loss_MSE(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1) )   
        return reconstruction_loss

    def compute_loss_MSE_VAE(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)       
        vae_reconst_loss = tf.reduce_mean(reconstruction_loss)
        vae_kl_loss = tf.reduce_mean(kl_loss)
        
        return vae_loss, vae_reconst_loss, vae_kl_loss

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


def train_vae(alpha=1e-3, steps=2, epochs=5, batch_size=64, write_summary=True):
    tf.reset_default_graph()
    
    with tf.Session() as sess:

        global_step = tf.Variable(0, name='global_step', trainable=False)
    
        log_main_path = './'+logdir_name
        if not os.path.isdir(log_main_path):
            os.mkdir(log_main_path)
        dirs = [os.path.join(log_main_path, o) for o in os.listdir(log_main_path) if os.path.isdir(os.path.join(log_main_path,o))]
        run_number = len(dirs)    
        
        if write_summary: 
            writer = tf.summary.FileWriter('{0}/run_{1:02.0f}'.format(logdir_name,run_number), sess.graph)
    
        network = Autoencoder()
        train_op = tf.train.AdamOptimizer(alpha).minimize(network.loss, global_step=global_step)
    #    train_op = tf.train.AdadeltaOptimizer().minimize(network.loss, global_step=global_step)
        tf.global_variables_initializer().run()
    
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
                
                for _ in range(epochs):
                    images = next(training_data)
                    _, loss_value, summary = sess.run([train_op, network.loss, network.merged],
                                                      feed_dict={network.image: images})
                    if write_summary:
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


def load_vae():

	graph = tf.Graph()
	with graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config, graph=graph)

		network = Autoencoder()
		init = tf.global_variables_initializer()
		sess.run(init)

		saver = tf.train.Saver(max_to_keep=1)
		training_data = data_iterator(batch_size=64)

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
		except:
			raise ImportError("Could not restore saved model")

		return sess, network


if __name__ == '__main__':
#    alpha_v = [1e-4,1e-5,1e-6,1e-7]
#    while True:
#        for alpha in alpha_v:
            alpha=10e-4
            train_vae(alpha=alpha, steps=10, epochs=300, batch_size=64, write_summary=True)
#    load_vae()
