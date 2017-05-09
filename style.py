'''
Adaptation of style transfer for use in ablative compression
Utilizes a single image for both style and content

Phillip Kuznetsov 2017
'''
import caffe
import numpy as np
import util

# weights for content loss
CAFFENET_WEIGHTS = {"content": {"conv4": 0.66,  'fc8': 0.33},
                    "style": {"conv1": 0.2,
                              "conv2": 0.2,
                              "conv3": 0.2,
                              "conv4": 0.2,
                              "conv5": 0.2}}
class StyleTransfer:
    def __init__(self, image_net, style_weight=1, content_weight=1, layer_weights=CAFFENET_WEIGHTS):
        self._check_layer_weights(layer_weights)
        self.image_net = image_net
        self.is_init = False
        self.layer_weights = layer_weights
        self.style_weight = style_weight
        self.content_weight = content_weight
    def _check_layer_weights(self, layer_weights):
        assert 'content' in layer_weights and 'style' in layer_weights
        content_layers = layer_weights['content'].keys()
        style_layers = layer_weights['style'].keys()
        content_set= set(content_layers)
        style_set= set(style_layers)
        # assert content_set.issubset(style_set)

    def init_image(self, image):
        ''' loads image from path and prepares it for style transfer'''
        # TODO finish
        self.image_layer_rep = self._feed_forward(image)
        self.is_init = True

    def init_matrix_file(self, path):
        ''' loads layer representations from path '''
        # TODO finish
        self.is_init = True

    def _check_init(self):
        ''' returns whether or not the file is ready '''
        if not self.is_init:
            raise RuntimeError('style transfer has not been initialized')

    def get_gradient(self, generated_image):
        ''' returns the gradient of the generated image with the image loaded '''
        self._check_init()
        gen_layer_rep = self._feed_forward(generated_image)
        content_layers = self.layer_weights['content']
        style_layers = self.layer_weights['style']
        if self.style_weight != 0:
            layers = self.layer_weights['style'].keys()
        else:
            layers = self.layer_weights['content'].keys()
        layers.sort(reverse=True)
        # set the first layer gradient to 0
        self.image_net.blobs[layers[0]].diff[:] = 0
        for i, layer in enumerate(layers):
            next_layer = 'data' if i == len(layers)-1 else layers[i+1]
            grad = self.image_net.blobs[layer].diff[0]
            if layer in content_layers and self.content_weight !=0:
                weight = content_layers[layer]
                grad += weight * self.content_weight * self._get_content_grad(gen_layer_rep, self.image_layer_rep, layer)

            if layer in style_layers and self.style_weight != 0:
                weight = style_layers[layer]
                grad += weight * self.style_weight * self._get_style_grad(gen_layer_rep, self.image_layer_rep, layer)

            if next_layer == 'data':
                grad = self._backprop(grad, start=layer, end=None)
            else:
                grad = self._backprop(grad, start=layer, end=next_layer)
        return grad


    def _feed_forward(self, image):
        ''' feed forwards the image and saves relevant layers '''
        content_layers = self.layer_weights['content']
        style_layers = self.layer_weights['style']
        content_set = set(content_layers.keys())
        style_set = set(style_layers.keys())
        if self.style_weight != 0 and self.content_weight != 0:
            layers = content_set.union(style_set)
        elif self.style_weight == 0:
            layers = content_set
        elif self.content_weight == 0:
            layers = style_set
        content_output = {}
        style_output = {}

        # forward
        self.image_net.forward(data=image)

        # peruse the layers for gold
        for layer in layers:
            content = self.image_net.blobs[layer].data[0].copy()
            content.reshape((content.shape[0], -1))
            content_output[layer] = content
            if layer in style_layers and self.style_weight != 0:
                style_output[layer] = util.gram(content)

        return {'content' : content_output, 'style' : style_output}

    def _get_style_grad(self, gen_layer_rep, image_layer_rep, layer):
        '''
        Return style loss gradient as defined in Gatys et. al.
        '''
        # grab the style of generated image
        image_content = image_layer_rep['content'][layer]
        input_gram = image_layer_rep['style'][layer]

        # grab the style of generated
        gen_content = gen_layer_rep['content'][layer]
        gen_gram = gen_layer_rep['style'][layer]

        # l2 norm derivative is just the difference
        diff = gen_gram -input_gram
        c = (gen_gram.shape[0] * gen_gram.shape[1])**-2
        grad = c * diff.dot(gen_content.transpose((1,0,2))) * (gen_content > 0) # gradient claculation according to fzliu/style-transfer/style.py L110
        return grad

    def _backprop(self, grad, start, end):
        ''' backprops the gradient through the imagenet network '''
        # backprop back
        dst = self.image_net.blobs[start]

        dst.diff[...] = grad
        self.image_net.backward(start=start, end=end)

        # assumes that the input layer of the net is 'data'
        g = self.image_net.blobs['data'].diff.copy()

        dst.diff.fill(0.)   # reset objective after each step
        return g
    def _get_content_grad(self, gen_layer_rep, image_layer_rep, layer):
        '''
        Return content loss gradient as defined in Gatys et. al.
        '''

        # grab the style of generated image
        image_content = image_layer_rep['content'][layer]

        # grab the style of generated
        gen_content = gen_layer_rep['content'][layer]
        # l2 norm derivative is just the difference
        diff = image_content - gen_content
        grad = diff * (gen_content > 0) # gradient claculation according to fzliu/style-transfer/style.py L114
        return grad
