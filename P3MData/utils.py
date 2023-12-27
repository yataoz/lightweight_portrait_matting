import torch
import tensorflow as tf
import tensorpack as tp
from tensorpack.dataflow.base import DataFlowReentrantGuard, DataFlow, ProxyDataFlow
from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.format import LMDBData
from tensorpack.dataflow.serialize import _reset_df_and_get_size
from tensorpack.dataflow.remote import zmq, DIE, get_tqdm_kwargs
from tensorpack.dataflow.remote import dumps as remote_dumps, loads as remote_loads
from tensorpack.utils.utils import get_rng, logger, get_tqdm
import platform
import numpy as np
import cv2
import random
import os
import threading
import time
import tqdm
import zmq
from collections import namedtuple, deque, defaultdict
from multiprocessing import Process

import pdb

class TP_Dataflow_to_PT_Dataset(torch.utils.data.IterableDataset):
    """
    convert tensorpack dataflow to pytorch Dataset
    """
    def __init__(self, df):
        self.df = df
    
    def __iter__(self):
        self.df.reset_state()
        yield from self.df


class RandomSwitch(DataFlow):
    def __init__(self, df_list, chance_list):
        self.df_list = df_list
        self.num_dfs = len(self.df_list)
        assert len(chance_list) == self.num_dfs
        self.chance_list = chance_list
        
    def reset_state(self):
        for d in set(self.df_list):
            d.reset_state()

    def __iter__(self):
        iter_objs = [iter(d) for d in self.df_list]
        ind_list = np.arange(self.num_dfs)
        while 1:
            # random choose which dataflow to use
            which = np.random.choice(ind_list, p=self.chance_list)

            # generate data
            try:
                dp = next(iter_objs[which])
                if dp is not None:
                    #print("source={}".format(dp['source']))
                    yield dp
            except StopIteration:
                #print("call reset_state() of df_list[{}], {}".format(which, df_list[which]))
                self.df_list[which].reset_state()

class CountedCacheData(ProxyDataFlow):
    """
    modified from tp.dataflow.CacheData
    """
    def __init__(self, ds, buffer_size, num_reuse, random_read=False):
        self.buffer_size = buffer_size
        self.num_reuse = num_reuse
        self.random_read = random_read
        super(CountedCacheData, self).__init__(ds)

    def reset_state(self):
        super(CountedCacheData, self).reset_state()
        self._guard = DataFlowReentrantGuard()
        if self.random_read:
            self.rng = get_rng(self)
        self.buffer = []

    def __iter__(self):
        itr = self.ds.__iter__()
        with self._guard:
            while 1:
                if len(self.buffer) < self.buffer_size:
                    #print("generating from dataflow. buffer len={}".format(len(self.buffer)))
                    dp = next(itr)
                    # add new data point
                    self.buffer.append({'data': dp, 'use_count': 1})
            
                else: 
                    #print("generating from cache. buffer len={}".format(len(self.buffer)))
                    if self.random_read:
                        idx = self.rng.randint(0, len(self.buffer) - 1)
                    else:
                        idx = 0
                    itm = self.buffer[idx]
                    itm['use_count'] += 1
                    dp = itm['data']

                    # remove data point if use times > num_reuse
                    if itm['use_count'] == self.num_reuse:
                        del self.buffer[idx]

                yield dp

def send_dataflow_to_multi_addrs(df, addrs, zmq_socket_type=zmq.PUB, hwm=50, format=None, bind=True):
    """
    modified from tp.dataflow.remote.send_dataflow_zmq()
    Instead of taking one address, here it takes multiple addresses and send the same df to these addresses
    Socket type is PUB instead of PUSH
    """
    # if sender is zmq.PUB, then receiver must be zmq.SUB
    # if sender is zmq.PUSH, then receiver must be zmq.PULL
    assert zmq_socket_type in [zmq.PUB, zmq.PUSH]

    assert format in [None, 'zmq_op', 'zmq_ops']
    if format is None:
        dump_fn = remote_dumps
    else:
        from zmq_ops import dump_arrays
        dump_fn = dump_arrays

    ctx = zmq.Context()
    sockets = []
    for addr in addrs:
        socket = ctx.socket(zmq_socket_type)
        socket.set_hwm(hwm)
        if bind:
            socket.bind(addr)
        else:
            socket.connect(addr)
        sockets.append(socket)

    try:
        df.reset_state()
        for addr in addrs:
            logger.info("Serving data to {} with {} format ...".format(
                addr, 'default' if format is None else 'zmq_ops'))
        INTERVAL = 200
        q = deque(maxlen=INTERVAL)

        try:
            total = len(df)
        except NotImplementedError:
            total = 0
        tqdm_args = get_tqdm_kwargs(leave=True, smoothing=0.8)
        tqdm_args['bar_format'] = tqdm_args['bar_format'] + "{postfix}"
        while True:
            with tqdm.trange(total, **tqdm_args) as pbar:
                for dp in df:
                    start = time.time()
                    dump_data = dump_fn(dp)
                    for i, socket in enumerate(sockets):
                        socket.send(dump_data, copy=False)
                    #sockets[1].send(dump_data, copy=False)
                    q.append(time.time() - start)
                    pbar.update(1)
                    if pbar.n % INTERVAL == 0:
                        avg = "{:.3f}".format(sum(q) / len(q))
                        pbar.set_postfix({'AvgSendLat': avg})
    finally:
        logger.info("Exiting send_dataflow_zmq ...")
        for socket in sockets:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
        if not ctx.closed:
            ctx.destroy(0)

class RemoteDataZMQ(tp.dataflow.DataFlow):
    """
    modified from tp.dataflow.remote.RemoteDataZMQ()
    Socket type is SUB instead of PULL
    """
    def __init__(self, addr, zmq_socket_type=zmq.SUB, hwm=50, bind=False):
        # if sender is zmq.PUB, then receiver must be zmq.SUB
        # if sender is zmq.PUSH, then receiver must be zmq.PULL
        assert zmq_socket_type in [zmq.SUB, zmq.PULL]
        self._zmq_socket_type = zmq_socket_type

        assert addr
        self._addr = addr
        self._hwm = int(hwm)
        self._guard = DataFlowReentrantGuard()
        self._bind = bind

    def reset_state(self):
        self.cnt = 0

    def bind_or_connect(self, socket, addr):
        if self._bind:
            socket.bind(addr)
        else:
            socket.connect(addr)

    def __iter__(self):
        with self._guard:
            try:
                ctx = zmq.Context()
                socket = ctx.socket(self._zmq_socket_type)
                if self._zmq_socket_type == zmq.SUB:
                    socket.setsockopt_string(zmq.SUBSCRIBE, "")
                socket.set_hwm(self._hwm)
                self.bind_or_connect(socket, self._addr)

                while True:
                    dp = remote_loads(socket.recv(copy=False))
                    yield dp
                    self.cnt += 1
            finally:
                ctx.destroy(linger=0)