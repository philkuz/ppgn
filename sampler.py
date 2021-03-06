#!/usr/bin/env python
'''
Anh Nguyen <anh.ng8@gmail.com>
2016
'''
from __future__ import print_function
import os, sys
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
from scipy.ndimage.filters import gaussian_laplace as laplace_filter
import util
from style import StyleTransfer

class Sampler(object):

    def load_image(self, shape, path, output_dir='', save=True):
        ''' loads an image in bgr format '''
        images = np.zeros(shape,  dtype='float32')
        image_size = shape[2:]
        in_image = scipy.misc.imread(path)
        in_image = scipy.misc.imresize(in_image, (image_size[0], image_size[1]))
        images[0] = np.transpose(in_image, (2, 0, 1))   # convert to (3, 227, 227) format

        data = images[:,::-1]   # convert from RGB to BGR
        if save:
            name = "%s/samples/%s.jpg" % (output_dir, 'start')
            util.save_image(data, name)
        return data

    def get_code(self, encoder, data, layer, output_dir='', mask=None):
        '''
        Push the given image through an encoder (here, AlexNet) to get a code.
        '''

        # set up the inputs for the net:
        image_size = encoder.blobs['data'].shape[2:]    # (1, 3, 227, 227)

        # subtract the ImageNet mean
        image_mean = scipy.io.loadmat('misc/ilsvrc_2012_mean.mat')['image_mean'] # (256, 256, 3)
        topleft = self.compute_topleft(image_size, image_mean.shape[:2])
        image_mean = image_mean[topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]   # crop the image mean
        data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0)    # mean is already BGR
        # initialize the encoder
        encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)

        # extract the features
        if mask is not None:
            encoder.forward(data=data*mask)
        else:
            encoder.forward(data=data)
        features = encoder.blobs[layer].data.copy()

        return features


    def backward_from_x_to_h(self, generator, diff, start, end):
        '''
        Backpropagate the gradient from the image (start) back to the latent space (end) of the generator network.
        '''
        dst = generator.blobs[end]

        dst.diff[...] = diff
        generator.backward(start=end)
        g = generator.blobs[start].diff.copy()

        dst.diff.fill(0.)   # reset objective after each step

        return g


    def compute_topleft(self, input_size, output_size):
        '''
        Compute the offsets (top, left) to crop the output image if its size does not match that of the input image.
        The output size is fixed at 256 x 256 as the generator network is trained on 256 x 256 images.
        However, the input size often changes depending on the network.
        '''

        assert len(input_size) == 2, "input_size must be (h, w)"
        assert len(output_size) == 2, "output_size must be (h, w)"

        topleft = ((output_size[0] - input_size[0])/2, (output_size[1] - input_size[1])/2)
        return topleft


    def h_autoencoder_grad(self, h, encoder, decoder, gen_out_layer, topleft, mask=None, input_image=None):
        '''
        Compute the gradient of the energy of P(input) wrt input, which is given by decode(encode(input))-input {see Alain & Bengio, 2014}.
        Specifically, we compute E(G(h)) - h.
        Note: this is an "upside down" auto-encoder for h that goes h -> x -> h with G modeling h -> x and E modeling x -> h.
        '''
        generated = encoder.forward(feat=h)
        x0 = encoder.blobs[gen_out_layer].data.copy()    # 256x256

        # Crop from 256x256 to 227x227
        image_size = decoder.blobs['data'].shape    # (1, 3, 227, 227)
        cropped_x0 = x0[:,:,topleft[0]:topleft[0]+image_size[2], topleft[1]:topleft[1]+image_size[3]]

        if mask is not None:
            cropped_x0 = mask * input_image + (1 - mask) * cropped_x0

        # Push this 227x227 image through net
        decoder.forward(data=cropped_x0)
        code = decoder.blobs['fc6'].data

        g = code - h
        return g

    def get_edge_gradient(self, input_image, generated_image, edge_detector):
        ''' Return edge gradient'''
        # calculate the edges of the images
        input_edge = edge_detector.forward(data=input_image)['laplace'].copy()
        generated_edge = edge_detector.forward(data=generated_image)['laplace'].copy()
        # l2 norm derivative is just the difference
        diff = input_edge - generated_edge
        # backprop thru
        dst = edge_detector.blobs['laplace']

        dst.diff[...] = diff
        edge_detector.backward(start='laplace')
        g = edge_detector.blobs['data'].diff.copy()

        dst.diff.fill(0.)   # reset objective after each step
        return g

    # def sampling( self, condition_net, image_encoder, image_net, image_generator, edge_detector,
    #             gen_in_layer, gen_out_layer, start_code, content_layer,
    #             n_iters, lr, lr_end, threshold,
    #             layer, conditions, mask_inner=None, , input_image=None, #units=None, xy=0,
    #             epsilon1=1, epsilon2=1, epsilon3=1e-10,
    #             mask_epsilon=1e-6, edge_epsilon=1e-8,
    #             style_epsilon=1e-8, content_epsilon=1e-8,
    #             output_dir=None, reset_every=0, save_every=1):
    def sampling( self, condition_net, image_encoder, image_net, image_generator, edge_detector,
                gen_in_layer, gen_out_layer, start_code, content_layer,
                n_iters, lr, lr_end, threshold,
                layer, conditions, mask=None, input_image=None, #units=None, xy=0,
                epsilon1=1, epsilon2=1, epsilon3=1e-10,
                mask_epsilon=1e-6, edge_epsilon=1e-8,
                style_epsilon=1e-8, content_epsilon=1e-8,
                output_dir=None, reset_every=0, save_every=1):
        # Get the input and output sizes
        image_shape = condition_net.blobs['data'].data.shape
        generator_output_shape = image_generator.blobs[gen_out_layer].data.shape
        encoder_input_shape = image_encoder.blobs['data'].data.shape

        # Calculate the difference between the input image of the condition net
        # and the output image from the generator
        image_size = util.get_image_size(image_shape)
        generator_output_size = util.get_image_size(generator_output_shape)
        encoder_input_size = util.get_image_size(encoder_input_shape)

        # The top left offset to crop the output image to get a 227x227 image
        topleft = self.compute_topleft(image_size, generator_output_size)
        topleft_DAE = self.compute_topleft(encoder_input_size, generator_output_size)

        src = image_generator.blobs[gen_in_layer]     # the input feature layer of the generator

        # Make sure the layer size and initial vector size match
        assert src.data.shape == start_code.shape
        use_style_transfer= style_epsilon != 0 or content_epsilon !=0
        # setup style transfer
        if input_image is not None and use_style_transfer:
            style_transfer = StyleTransfer(image_net, style_weight=style_epsilon, content_weight=content_epsilon)
            style_transfer.init_image(input_image)
        elif input_image is None:
            # TODO setup loading the vector components
            raise NotImplementedError('input image must not be None')
        elif not use_style_transfer:
            print('not using style transfer')
        # Variables to store the best sample
        last_xx = np.zeros(image_shape)    # best image
        last_prob = -sys.maxint                 # highest probability

        h = start_code.copy()

        condition_idx = 0
        list_samples = []
        i = 0

        while True:
            step_size = lr + ((lr_end - lr) * i) / n_iters
            condition = conditions[condition_idx]  # Select a class
            # 1. Compute the epsilon1 term ---
            # : compute gradient d log(p(h)) / dh per DAE results in Alain & Bengio 2014
            d_prior = self.h_autoencoder_grad(h=h, encoder=image_generator, decoder=image_encoder, gen_out_layer=gen_out_layer, topleft=topleft_DAE, mask=mask, input_image=input_image)

            # 2. Compute the epsilon2 term ---
            # Push the code through the generator to get an image x
            image_generator.blobs["feat"].data[:] = h
            generated = image_generator.forward()
            x = generated[gen_out_layer].copy()       # 256x256

            # Crop from 256x256 to 227x227
            cropped_x_nomask = x[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
            if mask is not None:
                cropped_x = mask * input_image + (1 - mask) * cropped_x_nomask
            else:
                cropped_x = cropped_x_nomask

            # Forward pass the image x to the condition net up to an unit k at the given layer
            # Backprop the gradient through the condition net to the image layer to get a gradient image
            d_condition_x, prob, info = self.forward_backward_from_x_to_condition(net=condition_net, end=layer, image=cropped_x, condition=condition)

            if mask is not None:
                generated_image = (1 - mask) * d_condition_x
            else:
                generated_image = d_condition_x
            d_edge = self.get_edge_gradient(input_image, generated_image, edge_detector)
            d_condition_x = epsilon2 * generated_image + edge_epsilon * d_edge
            if use_style_transfer:
                d_condition_x += style_transfer.get_gradient(generated_image)
            if mask is not None:
                d_condition_x += mask_epsilon * (mask) * (input_image - cropped_x_nomask)

            # Put the gradient back in the 256x256 format
            d_condition_x256 = np.zeros_like(x)
            d_condition_x256[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = d_condition_x.copy()

            # Backpropagate the above gradient all the way to h (through generator)
            # This gradient 'd_condition' is d log(p(y|h)) / dh (the epsilon2 term in Eq. 11 in the paper)
            d_condition = self.backward_from_x_to_h(generator=image_generator, diff=d_condition_x256, start=gen_in_layer, end=gen_out_layer)
            # if i % 10 == 0:
            #     self.print_progress(i, info, condition, prob, d_condition)

            # 3. Compute the epsilon3 term ---
            noise = np.zeros_like(h)
            if epsilon3 > 0:
                noise = np.random.normal(0, epsilon3, h.shape)  # Gaussian noise

            d_h = epsilon1 * d_prior
            d_h += d_condition
            d_h += noise
            h += step_size/np.abs(d_h).mean() * d_h

            h = np.clip(h, a_min=0, a_max=30)   # Keep the code within a realistic range

            # Reset the code every N iters (for diversity when running a long sampling chain)
            if reset_every > 0 and i % reset_every == 0 and i > 0:
                h = np.random.normal(0, 1, h.shape)

                # Experimental: For sample diversity, it's a good idea to randomly pick epsilon1 as well
                epsilon1 = np.random.uniform(low=1e-6, high=1e-2)

            # Save every sample
            last_xx = cropped_x.copy()
            last_prob = prob

            # Filter samples based on threshold or every N iterations
            if save_every > 0 and i % save_every == 0 and prob > threshold:
                name = "%s/samples/%05d.jpg" % (output_dir, i)

                label = self.get_label(condition)
                if mask is not None:
                    image = last_xx * mask + (1 - mask) * input_image
                else:
                    image = last_xx
                # TODO check why this wasn't the case
                # list_samples.append( (last_xx.copy(), name, label) )
                list_samples.append( (image.copy(), name, label) )

            # Stop if grad is 0
            if norm(d_h) == 0:
                print(" d_h is 0")
                break

            # Randomly sample a class every N iterations
            if i > 0 and i % n_iters == 0:
                condition_idx += 1

                if condition_idx == len(conditions):
                    break

            i += 1  # Next iter

        # returning the last sample
        print( "-------------------------")
        print("Last sample: prob [%s] " % last_prob)
        # if mask is not None:
        #     return last_xx * mask + (1 - mask) * input_image, list_samples
        return last_xx, list_samples
