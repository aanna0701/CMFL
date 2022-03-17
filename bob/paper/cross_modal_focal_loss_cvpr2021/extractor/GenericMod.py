import numpy as np 
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.learn.pytorch")

class GenericExtractorMod(Extractor):
	""" The class implementing a generic image extractor

	Attributes
	----------
	network: :py:class:`torch.nn.Module`
			The network architecture
	transforms: :py:mod:`torchvision.transforms`
			The transform from numpy.array to torch.Tensor

	"""
	
	def __init__(self, network, extractor_function, transforms = transforms.Compose([transforms.ToTensor()]), extractor_file=None,**kwargs):

																 
		""" Init method

			`extractor_function` the function which provides the final score from the network output (TODO: May be move the network to config too)
			`network` initialized architecture ; which is ready to load a pretrained model


		Parameters
		----------
		model_file: str
			The path of the trained PAD network to load
		transforms: :py:mod:`torchvision.transforms` 
			Tranform to be applied on the image
		scoring_method: str
			The scoring method to be used to get the final score, 
			available methods are ['pixel_mean','binary','combined']. 
		
		"""

		Extractor.__init__(self, skip_extractor_training=True)
		
		# model
		self.transforms = transforms 
		self.network = network
		self.extractor_function=extractor_function
		self.kwargs=kwargs

		print("extractor_file",extractor_file)

		self.load(extractor_file)




	def load(self, extractor_file):
		"""Loads the parameters required for feature extraction from the extractor file.
		This function usually is only useful in combination with the :py:meth:`train` function.
		In this base class implementation, it does nothing.

		**Parameters:**

		extractor_file : str
			The file to read the extractor from.
		"""
		if extractor_file is None or '.hdf5' in extractor_file :
			# do nothing (used mainly for unit testing) 
			logger.debug("No pretrained file provided in the config, will try loading the commandline path!")
			pass
		else:


			# With the new training
			logger.debug('Starting to load the pretrained PAD model')

			try:
				cp = torch.load(extractor_file,map_location=lambda storage,loc:storage)
				logger.debug('Loaded the model from cp.')
			except:
				raise ValueError('Could not load pretrained model : {}'.format(extractor_file))

			if 'state_dict' in cp:
				self.network.load_state_dict(cp['state_dict'])
				logger.debug('Loaded the model from state_dict.')
			else: ## check this part
				try:
					print('using the other way')
					self.network.load_state_dict(cp)
					logger.debug('Loaded the model from cp directly.')
				except:
					raise ValueError('Could not load the state dict : {}'.format(extractor_file))

			logger.debug('Loaded the pretrained PAD model')    

		self.network.eval()



	def __call__(self, image):
		""" Extract features from an image

		Parameters
		----------
		image : 3D :py:class:`numpy.ndarray`
			The image to extract the score from. Its size must be 3x224x224;
			
		Returns
		-------
		output : float
			The extracted feature is a scalar values ~1 for bonafide and ~0 for PAs
		
		"""

		input_image = np.rollaxis(np.rollaxis(image, 2),2) # changes to 128x128xnum_channels
		input_image = self.transforms(input_image)
		input_image = input_image.unsqueeze(0)

		output = self.network.forward(Variable(input_image))
	 
		
		score=self.extractor_function(output,self.kwargs)


		# output is a scalar score

		return score



