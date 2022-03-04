# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np

import PyTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from time import time

cuda.init()

class GPU():
	None

class CommonTsetlinMachine():
	def __init__(self, number_of_clauses, T, s, clause_drop_p=0.0, feature_drop_p=0.0, number_of_gpus=1, q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.number_of_gpus = np.minimum(cuda.Device.count(), number_of_gpus)

		self.number_of_clauses = number_of_clauses
		self.number_of_clauses_multi = int(number_of_clauses // self.number_of_gpus)
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.q = q
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.grid = grid
		self.block = block
		
		self.clause_drop_p = clause_drop_p
		self.feature_drop_p = feature_drop_p

		self.X_train = np.array([])
		self.Y_train = np.array([])
		self.X_test = np.array([])
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		self.initialized = False

		self.gpus = []
		for c in range(self.number_of_gpus):
			print("Preparing GPU #%d" % (c))
			gpu = GPU()
			gpu.device_id = c
			gpu.device = cuda.Device(c)
			gpu.context = gpu.device.make_context()
			gpu.g = curandom.XORWOWRandomNumberGenerator() 

			gpu.mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
			gpu.prepare_encode = gpu.mod_encode.get_function("prepare_encode")
			gpu.encode = gpu.mod_encode.get_function("encode")

			self.gpus.append(gpu)

			gpu.context.pop()
		print()

	def encode_X_multi(self, X, encoded_X_gpus):
		number_of_examples = X.shape[0]

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		X_gpus = []
		for gpu in self.gpus:
			gpu.context.push()
			encoded_X_gpu = encoded_X_gpus[gpu.device_id-1]
			X_gpus.append(cuda.mem_alloc(Xm.nbytes))
			cuda.memcpy_htod_async(X_gpus[-1], Xm)
			gpu.context.pop()

		for i in range(len(self.gpus)):
			gpu = self.gpus[i]

			gpu.context.push()
			gpu.context.synchronize()
			
			encoded_X_gpu = encoded_X_gpus[i]
			X_gpu = X_gpus[i]

			if self.append_negated:			
				gpu.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			else:
				gpu.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			gpu.context.pop()

		for i in range(len(self.gpus)):
			gpu = self.gpus[i]

			gpu.context.push()
			gpu.context.synchronize()
			
			encoded_X_gpu = encoded_X_gpus[i]
			X_gpu = X_gpus[i]

			if self.append_negated:			
				gpu.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			else:
				gpu.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			gpu.context.pop()

		self.sync_gpus() ## NOT NEEDED

	def encode_X(self, X, encoded_X_gpu):
		number_of_examples = X.shape[0]

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		X_gpu = cuda.mem_alloc(Xm.nbytes)
		cuda.memcpy_htod(X_gpu, Xm)
		if self.append_negated:			
			self.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
			self.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		else:
			self.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
			self.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()

	def allocate_gpu_memory_multi(self, number_of_examples):
		for gpu in self.gpus:
			gpu.context.push()
			gpu.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses_multi*self.number_of_ta_chunks*self.number_of_state_bits*4)
			gpu.clause_weights_gpu = cuda.mem_alloc(self.number_of_clauses_multi*4)
			gpu.class_sum_gpu = cuda.mem_alloc(self.number_of_classes*4)
			gpu.drop_clause_gpu = cuda.mem_alloc(self.number_of_clauses_multi*4)
			gpu.drop_feature_gpu = cuda.mem_alloc(self.number_of_ta_chunks*4)
			gpu.context.pop()


	def get_weight(self, mc_tm_class, clause):
		global_clause = mc_tm_class*self.number_of_clauses//self.number_of_classes + clause
		gpu_clause = global_clause % self.number_of_clauses_multi
		gpu_id = global_clause//self.number_of_clauses_multi

		return(self.get_state()[1][gpu_id][gpu_clause])

	def ta_action(self, mc_tm_class, clause, ta):
		state = self.get_state()[0]
		global_clause = mc_tm_class*self.number_of_clauses//self.number_of_classes + clause
		gpu_clause = global_clause % self.number_of_clauses_multi
		gpu_id = global_clause//self.number_of_clauses_multi
		ta_state = state[gpu_id].reshape((self.number_of_clauses_multi, self.number_of_ta_chunks, self.number_of_state_bits))

		return (ta_state[gpu_clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0

	def get_state(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.ta_state = []
			self.clause_weights = []
			for gpu in self.gpus:
				gpu.context.push()
				gpu.ta_state = np.empty(self.number_of_clauses_multi*self.number_of_ta_chunks*self.number_of_state_bits).astype(np.uint32)
				cuda.memcpy_dtoh(gpu.ta_state, gpu.ta_state_gpu)
				gpu.clause_weights = np.empty(self.number_of_clauses_multi).astype(np.int32)
				cuda.memcpy_dtoh(gpu.clause_weights, gpu.clause_weights_gpu)
				gpu.context.pop()

				self.ta_state.append(gpu.ta_state)
				self.clause_weights.append(gpu.clause_weights)
		return((self.ta_state, self.clause_weights, self.number_of_classes, self.number_of_clauses, self.number_of_features, self.dim, self.patch_dim, self.number_of_patches, self.number_of_state_bits, self.number_of_ta_chunks, self.append_negated, self.min_y, self.max_y))

	def set_state(self, state):
		self.number_of_classes = state[2]
		self.number_of_clauses = state[3]
		self.number_of_clauses_multi = self.number_of_clauses/self.number_of_gpus
		self.number_of_features = state[4]
		self.dim = state[5]
		self.patch_dim = state[6]
		self.number_of_patches = state[7]
		self.number_of_state_bits = state[8]
		self.number_of_ta_chunks = state[9]
		self.append_negated = state[10]
		self.min_y = state[11]
		self.max_y = state[12]
		
		for i in range(len(self.gpus)):
			gpu = self.gpus[i]
			gpu.context.push()
			gpu.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses_multi*self.number_of_ta_chunks*self.number_of_state_bits*4)
			gpu.clause_weights_gpu = cuda.mem_alloc(self.number_of_clauses_multi*4)
			cuda.memcpy_htod(gpu.ta_state_gpu, state[0][i])
			cuda.memcpy_htod(gpu.clause_weights_gpu, state[1][i])
			gpu.context.pop()

		self.X_train = np.array([])
		self.Y_train = np.array([])
		self.X_test = np.array([])
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

	def sync_gpus(self):
		for gpu in self.gpus:
			gpu.context.push()
			gpu.context.synchronize()
			gpu.context.pop()

	# Transform input data for processing at next layer
	def transform(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4))
		self.encode_X(X, encoded_X_gpu)

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d

#define NEGATIVE_CLAUSES %d

#define PATCHES %d

#define NUMBER_OF_EXAMPLES %d
		""" % (self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples)

		mod = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
		transform = mod.get_function("transform")

		X_transformed_gpu = cuda.mem_alloc(number_of_examples*self.number_of_clauses*4)
		transform(self.ta_state_gpu, encoded_X_gpu, X_transformed_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()
		X_transformed = np.ascontiguousarray(np.empty(number_of_examples*self.number_of_clauses, dtype=np.uint32))
		cuda.memcpy_dtoh(X_transformed, X_transformed_gpu)
		
		return X_transformed.reshape((number_of_examples, self.number_of_clauses))

	def _fit(self, X, encoded_Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		if (not self.initialized):
			self.initialized = True
						
			if len(X.shape) == 3:
				self.dim = (X.shape[1], X.shape[2],  1)
			elif len(X.shape) == 4:
				self.dim = X.shape[1:]

			if self.append_negated:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))*2
			else:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))

			self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1)*(self.dim[1] - self.patch_dim[1] + 1))
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
		
			parameters_multi = """
	#define CLASSES %d
	#define CLAUSES %d
	#define FEATURES %d
	#define STATE_BITS %d
	#define BOOST_TRUE_POSITIVE_FEEDBACK %d
	#define S %f
	#define THRESHOLD %d
	#define Q %f
	
	#define NEGATIVE_CLAUSES %d

	#define PATCHES %d

	#define NUMBER_OF_EXAMPLES %d
""" % (self.number_of_classes, self.number_of_clauses_multi, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.q, self.negative_clauses, self.number_of_patches, number_of_examples)

			for gpu in self.gpus:
				gpu.context.push()
				mod_prepare = SourceModule(parameters_multi + kernels.code_header + kernels.code_prepare, no_extern_c=True)
				gpu.prepare = mod_prepare.get_function("prepare")
				gpu.context.pop()

			self.allocate_gpu_memory_multi(number_of_examples)

			for gpu in self.gpus:
				gpu.context.push()
				
				new_mod_update = SourceModule(parameters_multi + kernels.code_header + kernels.code_update, no_extern_c=True)
				gpu.update = new_mod_update.get_function("update")
				gpu.update.prepare("PPPPPPPPi")

				gpu.evaluate_update = new_mod_update.get_function("evaluate")
				gpu.evaluate_update.prepare("PPPPPPi")

				gpu.context.pop()

			self.encoded_X_training_gpus = []
			for gpu in self.gpus:
				gpu.context.push()
				self.encoded_X_training_gpus.append(cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4)))
				gpu.context.pop()

			self.Y_gpus = []
			for gpu in self.gpus:
				gpu.context.push()
				self.Y_gpus.append(cuda.mem_alloc(encoded_Y.nbytes))
				gpu.context.pop()

			for gpu in self.gpus:
				gpu.context.push()
				gpu.prepare(gpu.g.state, gpu.ta_state_gpu, gpu.clause_weights_gpu, gpu.class_sum_gpu, grid=self.grid, block=self.block)
				gpu.context.pop()

			self.sync_gpus()

		if (not np.array_equal(self.X_train, X)) or (not np.array_equal(self.encoded_Y_train, encoded_Y)):
			self.X_train = X
			self.encoded_Y_train = encoded_Y

			self.encode_X_multi(X, self.encoded_X_training_gpus)

			for i in range(len(self.gpus)):
				gpu = self.gpus[i]
				gpu.context.push()
				cuda.memcpy_htod_async(self.Y_gpus[i], encoded_Y)
				gpu.context.pop()

			self.sync_gpus()

		if incremental == False:
			for gpu in self.gpus:
				gpu.context.push()
				gpu.prepare(gpu.g.state, gpu.ta_state_gpu, gpu.clause_weights_gpu, gpu.class_sum_gpu, grid=self.grid, block=self.block)
				gpu.context.pop()
			self.sync_gpus()

		for epoch in range(epochs):
			drop_feature = (np.random.rand(self.number_of_features) <= self.feature_drop_p).astype(np.uint32)
			drop_feature_chunk = np.zeros(self.number_of_ta_chunks).astype(np.uint32)
			for k in range(self.number_of_features):
				if drop_feature[k] == 1:
					ta_chunk = k // 32
					ta_pos = k % 32
					drop_feature_chunk[ta_chunk] |= (1 << ta_pos)

			for i in range(len(self.gpus)):
				gpu = self.gpus[i]
				gpu.context.push()
				drop_clause = np.ascontiguousarray(np.random.rand(self.number_of_clauses_multi) <= self.clause_drop_p).astype(np.uint32)
				cuda.memcpy_htod(gpu.drop_clause_gpu, drop_clause)
				cuda.memcpy_htod(gpu.drop_feature_gpu, drop_feature_chunk)
				gpu.context.pop()

			for e in range(number_of_examples):
				for i in range(len(self.gpus)):
					gpu = self.gpus[i]
					gpu.context.push()
					class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes)).astype(np.int32)
					cuda.memcpy_htod(gpu.class_sum_gpu, class_sum)

					gpu.evaluate_update.prepared_call(self.grid, self.block, gpu.ta_state_gpu, gpu.clause_weights_gpu, gpu.class_sum_gpu, gpu.drop_clause_gpu, gpu.drop_feature_gpu, self.encoded_X_training_gpus[i], np.int32(e))
					gpu.context.pop()

				global_class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes)).astype(np.int32)
				for i in range(len(self.gpus)):
					gpu = self.gpus[i]
					gpu.context.push()
					gpu.context.synchronize()
					local_class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes)).astype(np.int32)
					cuda.memcpy_dtoh(local_class_sum, gpu.class_sum_gpu)
					global_class_sum += local_class_sum
					gpu.context.pop()

				for i in range(len(self.gpus)):
					gpu = self.gpus[i]
					gpu.context.push()
					cuda.memcpy_htod(gpu.class_sum_gpu, global_class_sum)
					gpu.context.pop()

				for i in range(len(self.gpus)):
					gpu = self.gpus[i]
					gpu.context.push()
					gpu.context.synchronize()
					gpu.update.prepared_call(self.grid, self.block, gpu.g.state, gpu.ta_state_gpu, gpu.clause_weights_gpu, gpu.class_sum_gpu, gpu.drop_clause_gpu, gpu.drop_feature_gpu, self.encoded_X_training_gpus[i], self.Y_gpus[i], np.int32(e))
					gpu.context.pop()

				self.sync_gpus()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, X):
		number_of_examples = X.shape[0]
		
		if not np.array_equal(self.X_test, X):
			self.X_test = X

			self.encoded_X_test_gpus = []
			for gpu in self.gpus:
				gpu.context.push()
				self.encoded_X_test_gpus.append(cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4)))
				gpu.context.pop()

			self.encode_X_multi(X, self.encoded_X_test_gpus)

			parameters_multi = """
	#define CLASSES %d
	#define CLAUSES %d
	#define FEATURES %d
	#define STATE_BITS %d
	#define BOOST_TRUE_POSITIVE_FEEDBACK %d
	#define S %f
	#define THRESHOLD %d

	#define NEGATIVE_CLAUSES %d

	#define PATCHES %d

	#define NUMBER_OF_EXAMPLES %d
			""" % (self.number_of_classes, self.number_of_clauses_multi, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples)

			self.class_sum_gpus = []
			for gpu in self.gpus:
				gpu.context.push()
				mod = SourceModule(parameters_multi + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
				gpu.evaluate = mod.get_function("evaluate")
				local_class_sum = np.ascontiguousarray(np.empty(self.number_of_classes*number_of_examples)).astype(np.int32)
				self.class_sum_gpus.append(cuda.mem_alloc(local_class_sum.nbytes))
				gpu.context.pop()

		global_class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes*number_of_examples)).astype(np.int32)
		for i in range(len(self.gpus)):
			gpu = self.gpus[i]
			gpu.context.push()
			local_class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes*number_of_examples)).astype(np.int32)
			cuda.memcpy_htod(self.class_sum_gpus[i], local_class_sum)
			gpu.evaluate(gpu.ta_state_gpu, gpu.clause_weights_gpu, self.class_sum_gpus[i], self.encoded_X_test_gpus[i], grid=self.grid, block=self.block)
			gpu.context.pop()

		for i in range(len(self.gpus)):
			gpu = self.gpus[i]
			gpu.context.push()
			gpu.context.synchronize()
			cuda.memcpy_dtoh(local_class_sum, self.class_sum_gpus[i])
			global_class_sum += local_class_sum
			gpu.context.pop()
		
		class_sum_multi = np.clip(global_class_sum.reshape((self.number_of_classes, number_of_examples)), -self.T, self.T)

		return class_sum_multi
	
class MultiClassConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(self, number_of_clauses, T, s, patch_dim, number_of_gpus=1, q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, number_of_gpus=number_of_gpus, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		self.number_of_classes = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_classes), dtype = np.int32)
		for i in range(self.number_of_classes):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, X):
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class MultiClassTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, number_of_gpus=1, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, number_of_gpus=number_of_gpus, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_classes = 2#int(np.max(Y) + 1)
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_classes), dtype = np.int32)
		for i in range(self.number_of_classes):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, number_of_gpus=1, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, number_of_gpus=number_of_gpus, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_classes = 1
		self.patch_dim = (X.shape[1], 1, 1)
		
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return self.score(X) >= 0

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, number_of_gpus=1, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, number_of_gpus=number_of_gpus, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 0

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_classes = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y
