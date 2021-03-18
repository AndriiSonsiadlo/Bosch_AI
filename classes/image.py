#Copyright (C) 2020 BardzoFajnyZespol

import numpy as np

#image object for displaying anomaly results in Manual mode
class ImageObject:

	size = 0

	def __init__(self, path: str = '', status: bool = True, byteArray=[], mse: float = 1.0, index: int = -1):
		self.index = self.size
		self.__class__.size = self.__class__.size + 1
		self.path = path
		self.mse = mse
		self.status = status        # OK or NOK, TRUE or FALSE
		self.byteArray = byteArray
		self.byteNpArray = np.array(byteArray) / 255.0

		self.recon = None
		self.original = None
