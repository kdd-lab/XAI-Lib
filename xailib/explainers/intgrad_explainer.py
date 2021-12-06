from xailib.xailib_image import ImageExplainer
from xailib.models.bbox import AbstractBBox
import numpy as np
import pandas as pd
import tensorflow as tf

import torch

class IntgradImageExplainer(ImageExplainer):
    def __init__(self, bb:AbstractBBox):
        self.bb = bb
        super().__init__()
        
    def fit(self):
        return
    
    def explain(self, image, index_to_explain, baseline, preprocessing, predict, steps=50, model_type='tensorflow', cuda=False):
        """
        image: image to explain
        index_to_explain: Which class index of the prediciton to explain, if None the most probable will be selected
        baseline: baseline to use as reference. Possible choices are ['white', 'black', 'half', 'random']
            - white will use a white image as baseline
            - black will use a black one
            - half will average a white and a black one
            - random will use random color pixels
        preprocessing: function that takes as input an image and return the correct format for the black box
        predict: function that takes as input the output of preprocessing function and return an array of probabilities of the classes with shape (-1,num_classes)
        steps: number of images to use to produce the saliency map
        model_type: library used for your blackbox: tensorflow or pytorch 
        cuda: set to True if your model runs on GPU
        """
        
        self.preprocessing = preprocessing
        self.predict = predict
            
        if model_type=='tensorflow':
            attributions = self.tensorflow_explain(image=image, index_to_explain=index_to_explain, baseline=baseline, steps=steps, cuda=cuda)
        elif model_type=='pytorch':
            attributions = self.pytorch_explain(image=image, index_to_explain=index_to_explain, baseline=baseline, steps=steps, cuda=cuda)
        else:
            raise Exception('Model Type not Understood')
        return attributions

############################# TENSORFLOW #################################
    
    @staticmethod
    def interpolate_images(baseline, image, alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    @staticmethod
    def compute_gradients(images, target_class_idx, predict):
        with tf.GradientTape() as tape:
            tape.watch(images)
            probs = predict(images)[:, target_class_idx]
            return tape.gradient(probs, images)

    @staticmethod
    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients
    
    #@tf.function  # disable eager execution for faster run time, however using this decorator could lead to a memory leak if you create a newer model every iteration see https://github.com/tensorflow/tensorflow/issues/42441 for further information
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
                                                    predict=self.predict)
            # Write batch indices and gradients to extend TensorArray.
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()
        # 4. Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients)
        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (x - baseline) * avg_gradients
        return integrated_gradients

    def tensorflow_explain(self, image, index_to_explain, baseline, steps=50, cuda=False):
        # compute the integrated gradients 
        if baseline == 'white':
            attributions = self.compute_scores(self.preprocessing(image), index_to_explain, 
                                               baseline=self.preprocessing(np.zeros(shape=image.shape)+255), m_steps=steps, batch_size=32).numpy()
        elif baseline == 'black':
            attributions = self.compute_scores(self.preprocessing(image), index_to_explain, 
                                               baseline=self.preprocessing(np.zeros(shape=image.shape)+0), m_steps=steps, batch_size=32).numpy()
        elif baseline == 'half':
            attributions = []
            attributions.append(self.compute_scores(self.preprocessing(image), index_to_explain, 
                                                    baseline=self.preprocessing(np.zeros(shape=image.shape)+255), m_steps=steps, batch_size=32).numpy())
            attributions.append(self.compute_scores(self.preprocessing(image), index_to_explain, 
                                                    baseline=self.preprocessing(np.zeros(shape=image.shape)+0), m_steps=steps, batch_size=32).numpy())
            attributions = np.average(np.array(attributions), axis=0)
        elif baseline == 'random':
            attributions = []
            for i in range(10):
                attributions.append(self.compute_scores(self.preprocessing(image), index_to_explain, 
                                                    baseline=self.preprocessing(np.random.random_sample(image.shape)+255), m_steps=steps, batch_size=32).numpy())
            attributions = np.average(np.array(attributions), axis=0)
        else:
            raise Exception('baseline method not supported')
        return attributions
    
######################## PYTORCH ##########################

    def compute_outputs_and_gradients(self, inputs, target_label_idx, cuda=False):
        # do the pre-processing
        predict_idx = None
        gradients = []
        for input in inputs:
            with torch.autograd.set_grad_enabled(True):
                input = self.preprocessing(input)
                input.requires_grad=True
                output = self.predict(input)
                index = np.ones((output.size()[0], 1)) * target_label_idx
                index = torch.tensor(index, dtype=torch.int64)
                if cuda:
                    index = index.cuda()
                output = output.gather(1, index)
                # clear grad
                self.bb.zero_grad()
                gradient = torch.autograd.grad(torch.unbind(output), input)[0].detach().numpy()
                gradients.append(gradient)
        gradients = np.array(gradients)
        return gradients, target_label_idx

    def integrated_gradients(self, image, target_label_idx, baseline, steps=50, cuda=False):
        # scale inputs and compute gradients
        image = np.array(image)
        scaled_inputs = [baseline + (float(i) / steps) * (image - baseline) for i in range(0, steps + 1)]
        grads, _ = self.compute_outputs_and_gradients(scaled_inputs, target_label_idx, cuda)
        avg_grads = np.average(grads[:-1], axis=0).squeeze(0)
        avg_grads = np.transpose(avg_grads, (1, 2, 0))
        delta_X = (self.preprocessing(image) - self.preprocessing(baseline)).detach().squeeze(0).cpu().numpy()
        delta_X = np.transpose(delta_X, (1, 2, 0))
        integrated_grad = delta_X * avg_grads
        return integrated_grad

    def random_baseline(self, image, target_label_idx, steps, num_random_trials, cuda):
        all_intgrads = []
        for i in range(num_random_trials):
            integrated_grad = self.integrated_gradients(image, target_label_idx,
                                                        baseline=255.0*np.random.random_sample(image.shape), steps=steps, cuda=cuda)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads

    def white_and_black_baseline(self, image, target_label_idx, steps, cuda):
        all_intgrads = []
        integrated_grad = self.integrated_gradients(image, target_label_idx,
                                                    baseline=255.0+np.zeros(image.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        integrated_grad = self.integrated_gradients(image, target_label_idx, 
                                                    baseline=0.0+np.zeros(image.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads
    
    def pytorch_explain(self, image, index_to_explain, baseline, steps=50, cuda=False):
        
        # compute the integrated gradients 
        if baseline == 'white':
            attributions = self.integrated_gradients(image, index_to_explain, 
                                                     baseline=255.0*np.ones(image.shape), steps=steps, cuda=cuda)
        elif baseline == 'black':
            attributions = self.integrated_gradients(image, index_to_explain,
                                                     baseline=0.0*np.ones(image.shape), steps=steps, cuda=cuda)
        elif baseline == 'half':
            attributions = self.white_and_black_baseline(image, index_to_explain,
                                                         steps=steps, cuda=cuda)
        elif baseline == 'random':
            attributions = self.random_baseline(image, index_to_explain, 
                                                steps=steps, num_random_trials=10, cuda=cuda)
        else:
            raise Exception('baseline method not supported')
        return attributions

