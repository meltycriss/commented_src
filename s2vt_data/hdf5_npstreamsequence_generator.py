#!/usr/bin/env python

import h5py
import numpy as np
import os
import random
import sys

class SequenceGenerator():
  def __init__(self):
    self.dimension = 10
    self.batch_stream_length = 2000
    self.batch_num_streams = 8 # ???
    self.min_stream_length = 13
    self.max_stream_length = 17
    self.substream_names = None# ???
    self.streams_initialized = False

  def streams_exhausted(self):
    return False

  def init_streams(self):
    self.streams = [None] * self.batch_num_streams
    self.stream_indices = [0] * self.batch_num_streams
    self.reset_stream(0)
    self.streams_initialized = True

  # set substream_names to ordered streams's key
  # set/reset streams[stream_index] to a map <stream_name, content>
  # set stream_indices[stream_index] = 0

  ### result ###
  # in short read all info of the ith stream into streams[i]
  # self.substream become keys' name like 'cont_sentence', 'framefc7'
  # self.streams[stream_index] become the dictionary cont_sentence: xxx, framefc7:xxx
  # self.stream_indices[stream_index] = 0

  # Q: what's the necessity of setting self.streams[i] to the same data??? one copy is enough?
  # A: different streams have different content
  def reset_stream(self, stream_index):
    # streams: map <name, content>
    # get_streams() return a map. e.g. cont_sentence: xxx, input_sentence: xxx
    streams = self.get_streams() # ??? different from init_streams? map or list? streams != self.streams
    stream_names = sorted(streams.keys())
    # substream_names contains 'cont_sentence', 'input_sentence'
    if self.substream_names is None:
      assert len(stream_names) > 0
      self.substream_names = stream_names
    assert self.substream_names == stream_names

    # guarantee indexed stream exists
    if self.streams[stream_index] is None:
      self.streams[stream_index] = {}
    # guarantee all streams in the same length
    stream_length = len(streams[stream_names[0]])
    for k, v in streams.iteritems():
      if isinstance(v, np.ndarray):
        assert stream_length == v.shape[0]
      else:
        assert stream_length == len(v) 
      # streams[stream_index] is a map <stream_name, content>
      self.streams[stream_index][k] = v

    # mark indexed stream as 0 in stream_indices
    self.stream_indices[stream_index] = 0

  # Pad with zeroes by default -- override this to pad with soemthing else
  # for a particular stream
  def get_pad_value(self, stream_name):
    return 0

  # return batch and batch_indicator

  # batch is something like this
  #                     batch
  #     ---cont---  ---framefc7---  ---input---
  #     #0  #1  #2    #0  #1  #2    #0  #1  #2
  # t0
  # t1
  # t2

  # batch_indicator is something like this
  #     batch_indicator
  #     #0    #1    #2
  # t0
  # t1
  # t2
  def get_next_batch(self, truncate_at_exhaustion=True):
    if not self.streams_initialized:
      self.init_streams()
    # format: len0: [s0, s2, num of streams, s_n]
    #         len1: [s0, s2, num of streams, s_n]
    #         len2: [s0, s2, num of streams, s_n]
    batch_size = self.batch_num_streams * self.batch_stream_length
    batch = {}
    batch_indicators = np.zeros((self.batch_stream_length, self.batch_num_streams))
    # reshape batch[name] like batch_indicators
    # and set value to pad value
    for name in self.substream_names:
      # if value is high dimension
      if name in self.array_type_inputs.keys():
        dim = self.array_type_inputs[name]
        batch[name] = self.get_pad_value(name) * np.ones((self.batch_stream_length, self.batch_num_streams, dim))
      # if value is 1d
      else:
        # each batch[name] is a T * N * dim blob
        batch[name] = self.get_pad_value(name) * np.ones_like(batch_indicators)
    # indicators for end of each stream
    exhausted = [False] * self.batch_num_streams
    all_exhausted = False
    # used to indicate whether the program has once pass batch_num_streams 
    reached_exhaustion = False
    num_completed_streams = 0
    for t in range(self.batch_stream_length):
      all_exhausted = True
      for i in range(self.batch_num_streams):
        if not exhausted[i]:
          # never been initialized or come to the end of a stream
          if self.streams[i] is None or \
              self.stream_indices[i] == len(self.streams[i][self.substream_names[0]]): 
            self.stream_indices[i] = 0
            # Q: self.streams_exhausted() always return false, so the expression is meaningless?
            # A: derived class will override function streams_exhausted
            # reached_exhaustion is forever True after pass through all lines
            reached_exhaustion = reached_exhaustion or self.streams_exhausted()
            # exhausted[i] indicates the end of ith stream i.e. all lines in ith stream are read
            if reached_exhaustion: exhausted[i] = True
            # Q: why reset stream i? self.streams is the same data for all stream i
            # A: get_streams() in reset_stream() is wrapped around
            if not reached_exhaustion or not truncate_at_exhaustion:
              self.reset_stream(i)
            else:
              continue

          # import pdb; pdb.set_trace()
          # loading stream data into the corresponding sub_batch. e.g. batch[cont], batch[framefc7]
          for name in self.substream_names:
            if isinstance(self.streams[i][name], np.ndarray) and self.streams[i][name].ndim > 1:
              batch[name].resize((batch_size, self.streams[i][name].shape[1],1))
              batch[name][(t*self.batch_num_streams + i), :,0] = self.streams[i][name][self.stream_indices[i],:]
            elif name in self.array_type_inputs.keys():
              batch[name][t, i] = self.streams[i][name][self.stream_indices[i]][0,:]
            else:
              batch[name][t, i] = self.streams[i][name][self.stream_indices[i]]

          # batch_indicator is 0 in the begining of stream
          batch_indicators[t, i] = 0 if self.stream_indices[i] == 0 else 1
          # stream_indices[i] indicates the current index
          self.stream_indices[i] += 1
          if self.stream_indices[i] == len(self.streams[i][self.substream_names[0]]):
            num_completed_streams += 1
        if not exhausted[i]: all_exhausted = False

      if all_exhausted and truncate_at_exhaustion:
        print ('Exhausted all data; cutting off batch at timestep %d ' +
               'with %d streams completed') % (t, num_completed_streams)
        for name in self.substream_names:
          # content of batch[name] is storaged in batch[name]'s first t-1 row
          batch[name] = batch[name][:t, :]
        batch_indicators = batch_indicators[:t, :]
        break
    return batch, batch_indicators

  def get_streams(self):
    raise Exception('get_streams should be overridden to return a dict ' +
                    'of equal-length iterables.')

class HDF5SequenceWriter():
  def __init__(self, sequence_generator, output_dir=None, verbose=False):
    self.generator = sequence_generator
    assert output_dir is not None  # required
    self.output_dir = output_dir
    if os.path.exists(output_dir):
      raise Exception('Output directory already exists: ' + output_dir)
    os.makedirs(output_dir)
    self.verbose = verbose
    self.filenames = []

  def write_batch(self, stop_at_exhaustion=False):
    batch_comps, cont_indicators = self.generator.get_next_batch()
    batch_index = len(self.filenames)
    filename = '%s/batch_%d.h5' % (self.output_dir, batch_index)
    self.filenames.append(filename)
    # get the handler
    h5file = h5py.File(filename, 'w')
    # return the container
    dataset = h5file.create_dataset('cont', shape=cont_indicators.shape, dtype=cont_indicators.dtype)
    # write data intot the container
    dataset[:] = cont_indicators
    dataset = h5file.create_dataset('buffer_size', shape=(1,), dtype=np.int)
    dataset[:] = self.generator.batch_num_streams
    for key, batch in batch_comps.iteritems():
      if self.verbose:
        for s in range(self.generator.batch_num_streams):
          stream = np.array(self.generator.streams[s][key])
          print 'batch %d, stream %s, index %d: ' % (batch_index, key, s), stream
      h5dataset = h5file.create_dataset(key, shape=batch.shape, dtype=batch.dtype)
      h5dataset[:] = batch
    # write in
    h5file.close()

  def write_to_exhaustion(self):
    while not self.generator.streams_exhausted():
      self.write_batch(stop_at_exhaustion=True)

  def write_filelists(self):
    assert self.filenames is not None
    filelist_filename = '%s/hdf5_chunk_list.txt' % self.output_dir
    with open(filelist_filename, 'w') as listfile:
      for filename in self.filenames:
        listfile.write('%s\n' % filename)
