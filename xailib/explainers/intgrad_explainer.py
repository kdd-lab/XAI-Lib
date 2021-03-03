from xailib.xailib_image import ImageExplainer
from xailib.models.bbox import AbstractBBox
import pandas as pd
import tensorflow as tf
import numpy as np


class IntgradImageExplainer(ImageExplainer):
    random_state = 0
    bb = None  # The Black Box to be explained
        
    @staticmethod
    def interpolate_images(baseline, image, alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    @staticmethod
    def compute_gradients(images, target_class_idx, model):
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
            return tape.gradient(probs, images)

    @staticmethod
    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def __init__(self, bb: AbstractBBox):
        self.bb = bb
        self.config = {
            'baseline': None,
            'target_index': None
        }
        super().__init__()

    def fit(self, class_name, baseline='black', _df=None):
        self.config['baseline'] = baseline
        self.config['target_index'] = class_name

    #@tf.function  # disable eager execution for faster run time, however using this decorator could lead to a memory leak if you create a newer model evry iteration see https://github.com/tensorflow/tensorflow/issues/42441 for further information
    def compute_scores(self,
                       x,  # input image
                       target_class_idx,  # target class to generate the saliency map
                       baseline='black',
                       m_steps=50,
                       batch_size=32):
            
        # 1. Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
        
        # Initialize TensorArray outside loop to collect gradients.    
        gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
            
        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]
        
            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = self.interpolate_images(baseline=baseline,
                                                                    image=x,
                                                                    alphas=alpha_batch)
        
            # 3. Compute gradients between model outputs and interpolated inputs.
            gradient_batch = self.compute_gradients(images=interpolated_path_input_batch,
                                                    target_class_idx=target_class_idx,
                                                    model=self.bb)
            
            # Write batch indices and gradients to extend TensorArray.
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
        
        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()
        
        # 4. Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients)
        
        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (x - baseline) * avg_gradients
        
        return integrated_gradients

    def explain(self, x):
        """
        :param x: the image we want to explain, run fit before to assign baseline and the target class
        for whom you want to get the scores
        :return: explanation object
        """
        baseline = self.config['baseline']
        target_idx = self.config['target_index']

        if not target_idx:
            raise Exception(f'the target class to explain is not defined, pleas run the fit method')

        # check baseline argument is in the correct format
        if isinstance(baseline, str):
            if baseline == 'black':
                baseline = tf.zeros(shape=x.shape)
            elif baseline == 'white':
                baseline = tf.zeros(shape=x.shape)+255
            else:
                raise Exception(f'Baseline color {baseline} not recognized, options are: black, white')
        else:
            assert baseline.shape == x.shape, 'shape mismatch between baseline and input image'
            if isinstance(baseline, np.ndarray):
                baseline = tf.convert_to_tensor(baseline,dtype=tf.float32)
            else:
                assert tf.is_tensor(baseline), 'baseline must be a tensorflow tensor or a numpy array'
                baseline = tf.cast(baseline, tf.float32)

        # check if input image is in tensorflow or numpy
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        else:
            assert tf.is_tensor(x), 'x must be a tensorflow tensor or a numpy array'
            x = tf.cast(x, tf.float32)

        exp = self.compute_scores(x, target_idx, baseline, m_steps=50, batch_size=32).numpy()

        return exp
