from tvb.simulator.lab import *
# from tvb.datatypes import connectivity
# from tvb.simulator import integrators
# from tvb.simulator import coupling
import numpy as np
import numpy.random as rgn
import matplotlib.pyplot as plt
import math
import sys
import os

from numpy import corrcoef
import seaborn as sns

import time
import logging
import itertools
import argparse

import os, sys, inspect

from __future__ import division, print_function

from mpi4py import MPI
from regular_run import regularRun
import scipy.io as io
from pathlib import Path

comm = MPI.COMM_WORLD
my_rank = comm.rank
total_ranks = comm.size

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

sys.path.append("{}{}".format(parent_dir, '/NeuroML/'))

np.set_printoptions(threshold=sys.maxsize)

# for numexpr package missing and no permissions to install:
# clone package, copy to hpc, build with $ python setup.py build, copy numexpr folder from build/lib.linux-ppc64le-3.6 to project root
# and do: export PATH=$PATH:/\$PROJECT_cpcp0/vandervlag/all-benchmarking/numexpr/build/lib.linux-ppc64le-3.6/numexpr   // preferrable pythonpath
# same for tvb-data dependancy: move build directory (python setyp.py build) to root

# set global logger level in tvb.logger.library_logger.conf


rgn.seed(79)


class TVB_test:

	def __init__(self):
		self.sim_length = self.args.n_time # 400
		self.g = np.array([1.0])
		self.s = np.array([1.0])
		self.dt = 0.1
		self.period = 10.0
		self.omega = 60.0 * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(self.s, self.g, self.dt)
		self.integrator = integrators.EulerDeterministic(dt=self.dt)
		self.weights = self.SC = self.connectivity.weights
		self.lengths = self.connectivity.tract_lengths
		self.n_nodes = self.weights.shape[0]
		self.args = self.parse_args()
		self.tavg_period = 10.0
		self.nstep = self.args.n_time  # 4s
		self.n_inner_steps = int(self.tavg_period / self.dt)
		self.nc = self.args.n_coupling
		self.ns = self.args.n_speed
		self.couplings, self.speeds = self.setup_params(self.nc, self.ns)
		self.params = self.expand_params(self.couplings, self.speeds)
		self.n_work_items, self.n_params = self.params.shape
		self.min_speed = self.speeds.min()
		self.buf_len_ = ((self.lengths / self.min_speed / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of 2
		self.states = 1

	def tvb_connectivity(self, speed, global_coupling, dt=0.1):
		white_matter = connectivity.Connectivity.from_file(source_file="data/connectivity_68.zip")
		white_matter.configure()
		white_matter.speed = np.array([speed])
		white_matter_coupling = coupling.Linear(a=global_coupling)
		return white_matter, white_matter_coupling

	def tvb_python_model(self):
		whatmodel=self.args.model.lower()
		print(whatmodel)

		switcher = {
			'kuramoto': models.Kuramoto,
			'oscillator': models.Generic2dOscillator,
			'wongwang': models.ReducedWongWang,
			# 'montbrio': models.Montbrio,
			'epileptor': models.Epileptor
		}
		modelexe = switcher.get(whatmodel, 'invalid model')
		# print(modelexe)
		populations = modelexe()

		# populations = models.Kuramoto()
		populations.configure()
		populations.omega = np.array([self.omega])
		return populations

	def createFolder(directory):
		try:
			if not directory.exists():
				directory.mkdir()
		except OSError:
			print('Error: Creating directory. ' + directory)

	def generate_output(results, params_matrix, n_work_items):
		# Output
		sbj = '0/'
		npath1 = Path("/home/sandra/Documents/Thanos/TVB_MPI/")
		data_folder = npath1 / sbj
		print(data_folder)
		createFolder(data_folder)
		pathsaveFC = data_folder / "FC/"
		createFolder(pathsaveFC)
		pathsavetavg = data_folder / "TAVG/"
		createFolder(pathsavetavg)
		print(pathsavetavg)
		print(pathsaveFC)
		# createFolder(data_folder / "BOLD/")
		# pathsaveBOLD = data_folder / "BOLD/"
		# createFolder(data_folder / "FC_bold/")
		# pathsaveFC_bold = data_folder / "FC_bold/"

		tavg = np.array(results['tavg'])
		if 'bold' in results:
			bold = results['bold']
		# win_size = 20 #1  # 200  2s
		# print(tavg.shape)

		for j in range(n_work_items):
			ttmp = tavg[j]
			# win_tavg = tavg.reshape((tavg.shape[-1], win_size) + tavg.shape[1:])
			# cf1 = np.corrcoef(win_tavg[-1][:, :, j].T)
			cf1 = np.corrcoef(ttmp.T)
			io.savemat(str(pathsaveFC / "out_corr{:04d}".format(j)), {'correlation': cf1, 'param': params_matrix[j]})
			io.savemat(str(pathsavetavg / "out_tavg{:04d}".format(j)),
					   {'tavg': ttmp, 'param': params_matrix[j]})  # new tavg line
		# if 'bold' in results:
		#    io.savemat("{}out_bold{:04d}".format(pathsaveBOLD, j), {'bold': bold[:, :, j],'param': params_matrix[j]})  # new bold line
		#    io.savemat("{}out_corr_bold{:04d}".format(pathsaveFC_bold, j), {'corr_bold': cf2,'param': params_matrix[j]})  # new bold line

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')
		parser.add_argument('-c', '--n_coupling', help='num grid points for coupling parameter', default=32, type=int)
		parser.add_argument('-s', '--n_speed', help='num grid points for speed parameter', default=32, type=int)
		parser.add_argument('-t', '--test', help='check results', action='store_true')
		parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=400)
		parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true', default='-v')
		# parser.add_argument('-p', '--no_progress_bar', help='suppress progress bar', action='store_false')
		# parser.add_argument('--caching',
		# 					choices=['none', 'shared', 'shared_sync', 'shuffle'],
		# 					help="caching strategy for j_node loop (default shuffle)",
		# 					default='none'
		# 					)
		# parser.add_argument('--dataset',
		# 					choices=['hcp', 'sep'],
		# 					help="dataset to use (hcp: 100 nodes, sep: 645 nodes",
		# 					default='hcp'
		# 					)
		parser.add_argument('--node_threads', default=1, type=int)
		parser.add_argument('--model',
							choices=['Rwongwang', 'Kuramoto', 'Epileptor', 'Oscillator', \
									 'Oscillatorref', 'Kuramotoref', 'Rwongwangref'],
							help="neural mass model to be used during the simulation",
							default='Oscillator'
							)
		parser.add_argument('--lineinfo', default=True, action='store_true')

		parser.add_argument('--filename', default="kuramoto_network.c", type=str,
							help="Filename to use as GPU kernel definition")
		# parser.add_argument("bench", default="all", nargs='*', choices=["noop", "scatter", "gather", "all"], help="Which sub-set of kernel to run")

		parser.add_argument('-b', '--bench', default="regular", type=str, help="What to bench: regular, numba, cuda")

		args = parser.parse_args()
		return args

	def expand_params(self, couplings, speeds):  # {{{
		# the params array is transformed into a 2d array
		# by first creating tuples of (speed, coup) and arrayfying then
		# pycuda (check) threats them as flattenened arrays but numba needs 2d indexing
		params = itertools.product(speeds, couplings)
		params = np.array([vals for vals in params], np.float32)
		return params  # }}}

	def setup_params(self, nc, ns):  # {{{
		# the correctness checks at the end of the simulation
		# are matched to these parameter values, for the moment
		couplings = np.logspace(1.6, 3.0, nc)
		speeds = np.logspace(0.0, 2.0, ns)
		return couplings, speeds  # }}}

	def calculate_FC(self, timeseries):
		return corrcoef(timeseries.T)

	def correlation_SC_FC(self, SC, FC):
		return corrcoef(FC.ravel(), SC.ravel())[0, 1]

	def plot_SC_FC(self, SC, FC, tag):
		# print(FC)
		fig, ax = plt.subplots(ncols=2, figsize=(12, 3))
		sns.heatmap((FC), xticklabels='',
					yticklabels='', ax=ax[0],
					cmap='coolwarm')
		sns.heatmap(SC / SC.max(), xticklabels='', yticklabels='',
					ax=ax[1], cmap='coolwarm', vmin=0, vmax=1)  #
		r = self.correlation_SC_FC(SC, FC)
		ax[0].set_title('simulated FC. \n(SC-FC r = %1.4s )' % r)
		ax[1].set_title('SC')
		# plt.savefig("FC_SC_"+tag+".png")
		return r

	# Todo: check if this function work. derr_speed > 500 and derr_coupl < -1500 evaluate to false for pyCuda runs
	def check_results(self, n_nodes, n_work_items, tavg, weights, speeds, couplings, logger, args):
		r, c = np.triu_indices(n_nodes, 1)
		win_size = args.n_time  # 4s? orig 200 # 2s?
		win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
		err = np.zeros((len(win_tavg), n_work_items))
		logger.info('err.shape %s', err.shape)
		# TODO do cov/corr in kernel
		for i, tavg_ in enumerate(win_tavg):
			for j in range(n_work_items):
				fc = np.corrcoef(tavg_[:, :, j].T)
				# err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()   weights is 1 dim array
				# logger.info('fc[r, c].shape %s, weights[r].shape %s', fc[r, c].shape, weights[r].shape)
				err[i, j] = ((fc[r, c] - weights[r, c]) ** 2).sum()
		# look at 2nd 2s window (converges quickly)
		err_ = err[-1].reshape((speeds.size, couplings.size))
		# change on fc-sc metric wrt. speed & coupling strength
		derr_speed = np.diff(err_.mean(axis=1)).sum()
		derr_coupl = np.diff(err_.mean(axis=0)).sum()
		logger.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
		# if args.dataset == 'hcp':
		assert derr_speed > 350.0
		assert derr_coupl < -500.0
		# if args.dataset == 'sep':
		# 	assert derr_speed > 5e4
		# 	assert derr_coupl > 1e4

		logger.info('result OK')

	def regular(self, logger, wi, wi_per_rank, trace):
		logger.info('start regular TVB run')

		# Initialize Model
		model = self.tvb_python_model()
		# Initialize Monitors
		monitorsen = (monitors.TemporalAverage(period=self.period))
		# Initialize Simulator
		sim = simulator.Simulator(model=model, connectivity=self.connectivity, coupling=self.coupling,
								  integrator=self.integrator,
								  monitors=[monitorsen])
		sim.configure()

		if wi >= total_ranks:
			for i in range(0, wi_per_rank):
				speed, coupl = params[my_rank * wi_per_rank + i]
				# tvbRun = regularRun(n_time, coupl, speed, dt, period)
				# trace[i] = np.squeeze(tvbRun.simulate_python())

				(_, tavg_data) = sim.run(simulation_length=self.sim_length)[0]
				trace[i] = np.squeeze(tavg_data)

				print("Processed: " + str(i + my_rank * wi_per_rank))
		else:
			speed, coupl = params[my_rank + total_ranks * wi_per_rank]
			if my_rank <= wi:
				# tvbRun = regularRun(n_time, coupl, speed, dt, period)
				# trace[my_rank + total_ranks * wi_per_rank] = tvbRun.simulate_python()
				(_, tavg_data) = sim.run(simulation_length=self.sim_length)[0]
				trace[my_rank + total_ranks * wi_per_rank] = np.array(tavg_data)

		# print(np.squeeze(np.array(tavg_data)).shape)
		#
		# FC = self.calculate_FC(np.squeeze(np.array(tavg_data)))
		# r = self.plot_SC_FC(self.SC, FC,"regular")
		# print(r)
		# return np.squeeze(tavg_data)
		return trace

	def numba(self, logger, wi, wi_per_rank, trace):
		logger.info('start Numba run')
		from numbacuda_run import NumbaCudaRun
		numbacuda = NumbaCudaRun()
		trace = numbacuda.run_simulation(dt)

	#
	# (numbacuda_FC, python_r) = tvbhpc.simulate_numbacuda()
	# print(numbacuda_FC)
	# tavg_data = np.transpose(trace, (1, 2, 0))
	# tvbhpc.check_results(n_nodes, n_work_items, tavg_data, weights, speeds, couplings, logger, args)

	# numba kernel based on the c index used for cuda
	def numbac(self, logger, wi, wi_per_rank, trace):
		logger.info('start Numba run')
		from cindex_numbacuda_run import NumbaCudaRun
		numbacuda = NumbaCudaRun()

		threadsperblock = len(self.couplings)
		blockspergrid = len(self.speeds)
		logger.info('threadsperblock %d', threadsperblock)
		logger.info('blockspergrid %d', blockspergrid)

		tavg_data = numbacuda.run_simulation(blockspergrid, threadsperblock, self.n_inner_steps, self.n_nodes,
											 self.buf_len, self.dt, self.weights, self.lengths, self.params.T,
											 logger)
		logger.info('tavg_data.shape %s', tavg_data.shape)
		# logger.info('tavg_data %f', tavg_data)

	#
	# (numbacuda_FC, python_r) = tvbhpc.simulate_numbacuda()
	# print(numbacuda_FC)
	# tavg_data = np.transpose(trace, (1, 2, 0))
	# tvbhpc.check_results(n_nodes, n_work_items, tavg_data, weights, speeds, couplings, logger, args)

	def cuda(self, logger, wi, wi_per_rank, trace):
		logger.info('start Cuda run')

		# if wi >= total_ranks:
		# 	for i in range(0, wi_per_rank):
		# 		speed, coupl = params[my_rank * wi_per_rank + i]
		# 		# tvbRun = regularRun(n_time, coupl, speed, dt, period)
		# 		# trace[i] = np.squeeze(tvbRun.simulate_python())
		#
		# 		(_, tavg_data) = sim.run(simulation_length=self.sim_length)[0]
		# 		trace[i] = np.squeeze(tavg_data)
		#
		# 		print("Processed: " + str(i + my_rank * wi_per_rank))
		# else:
		# 	speed, coupl = params[my_rank + total_ranks * wi_per_rank]
		# 	if my_rank <= wi:
		# 		# tvbRun = regularRun(n_time, coupl, speed, dt, period)
		# 		# trace[my_rank + total_ranks * wi_per_rank] = tvbRun.simulate_python()
		# 		(_, tavg_data) = sim.run(simulation_length=self.sim_length)[0]
		# 		trace[my_rank + total_ranks * wi_per_rank] = np.array(tavg_data)

		from cuda_run import CudaRun
		cudarun = CudaRun()
		tavg_data = cudarun.run_simulation(self.weights, self.lengths, self.params, self.couplings, self.speeds, logger,
										   self.args, self.n_nodes, self.n_work_items, self.n_params, self.nstep,
										   self.n_inner_steps, self.buf_len, self.states, self.dt, self.min_speed)
		# logger.info('tavg_data %f', tavg_data)

		self.check_results(self.n_nodes, self.n_work_items, tavg_data, self.weights, self.speeds, self.couplings, logger, self.args)

		return tavg_data

	def startsim(self, pop, tmpld):

		tic = time.time()
		logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
		logger = logging.getLogger('[tvbBench.py]')

		if my_rank == 0:
			# TODO buf_len per speed/block
			logger.info('dt %f', self.dt)
			logger.info('nstep %d', self.nstep)
			# logger.info('caching strategy %r', self.args.caching)
			logger.info('n_inner_steps %f', self.n_inner_steps)
			if self.args.test and self.args.n_time % 200:
				logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing')  # }}}

			# setup data
			# weights = tvbhpc.connectivity.weights
			logger.info('weights.shape %s', self.weights.shape)
			# lengths = tvbhpc.connectivity.tract_lengths
			logger.info('lengths.shape %s', self.lengths.shape)
			# n_nodes = weights.shape[0]
			logger.info('n_nodes %d', self.n_nodes)
			logger.info('single connectome, %d x %d parameter space', self.ns, self.nc)
			logger.info('%d total num threads', self.ns * self.nc)
			logger.info('min_speed %f', self.min_speed)
			# buf_len = 2**np.argwhere(2**np.r_[:30] > buf_len_)[0][0]  # use next power of 2
			logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)

		tac = time.time()
		logger.info("Setup in: {}".format(tac - tic))

		benchwhat = self.args.bench

		self.args.filename = "{}{}{}{}".format(parent_dir, '/NeuroML/CUDAmodels/', self.args.model.lower(), '.c')

		if ('kuramoto' in self.args.model.lower()):
			self.states = 1
		elif 'oscillator' in self.args.model.lower():
			self.states = 2
		elif 'wongwang' in self.args.model.lower():
			self.states = 2
		elif 'montbrio' in self.args.model.lower():
			self.states = 2
		elif 'epileptor' in self.args.model.lower():
			self.states = 6
		logger.info('number of states %d', self.states)

		# mpi stuff
		wi = nc * ns
		wi_per_rank = 1
		if wi >= total_ranks:
			wi_per_rank = int(wi / total_ranks)
		extra = wi % total_ranks
		# params = [x for x in itertools.product(speeds, couplings)]

		if my_rank == 0:
			print("Work items: " + str(wi))
			print("Work items per rank: " + str(wi_per_rank))
			print("Extra: " + str(extra))

		my_trace = [None for i in range(wi_per_rank)]
		# TIME
		if my_rank == 0:
			t0 = time.time()
		# run_all(wi, wi_per_rank, extra, my_trace, speeds, couplings, nnode, args.n_time, dt, period, params)

			# locals()[benchwhat]()
			logger.info('benchwhat: %s', benchwhat)
			switcher = {
				'regular': self.regular,
				'numba': self.numba,
				'numbac': self.numbac,
				'cuda': self.cuda
			}
			run_rnc = switcher.get(benchwhat, 'invalid bench choice')

		run_rnc(logger, wi, wi_per_rank, my_trace)

		comm.Barrier()
		my_trace_g = np.array(comm.gather(my_trace, root=0))

		if my_rank == 0:
			trace = []
			for i in my_trace_g:
				for j in i:
					trace.append(j)

			t1 = time.time()
			total = t1 - t0
			print("Simulation time: " + str(total) + "\n")
			self.generate_output({'tavg': trace}, self.params, wi)
		comm.Barrier()

		if my_rank == 0:
			logger.info('filename %s', self.args.filename)
			logger.info('model %s', self.args.model)
			toc = time.time()
			print("Finished python simulation successfully in: {}".format(toc - tac))
			elapsed = toc - tic
			# inform about time
			logger.info('elapsed time %0.3f', elapsed)
			logger.info('%0.3f M step/s', 1e-6 * self.nstep * self.n_inner_steps * self.n_work_items / elapsed)
			logger.info('finished')


if __name__ == '__main__':
	zelf = TVB_test()
	zelf.startsim('Kuramoto', tmpld=0)
