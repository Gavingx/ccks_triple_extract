# -*- coding: utf-8 -*-
import os
import tempfile
import random
import json
import logging
from termcolor import colored
from albert_zh import modeling
from albert_zh import args
from albert_zh.args import PoolingStrategy
import contextlib


def import_tf(device_id=-1, verbose=False):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3' # edit by gavin: (0： 显示所有logs；1：隐藏 INFO logs；2：额外隐藏WARNING logs； 3：所有 ERROR logs也不显示)
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR) # edit by gavin: 将 TensorFlow 日志信息输出到屏幕(参数是日志等级）
    return tf


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def optimize_graph(logger=None, verbose=False, pooling_strategy=PoolingStrategy.REDUCE_MEAN, max_seq_len=40):
    if not logger:
        logger = set_logger(colored('BERT_VEC', 'yellow'), verbose) # edit by gavin: https://www.cnblogs.com/telecomshy/p/10630888.html
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(device_id=0, verbose=verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference # edit by gavin: optimize_for_inference 通过调用 optimize_for_inference 脚本，会自动删除模型中输入层和输出层之间所有不需要的节点，同时该脚本还做了一些其他优化以提高运行速度。例如它把显式批处理标准化运算跟卷积权重进行了合并，从而降低了计算量

        # allow_soft_placement:自动选择运行设备
        config = tf.ConfigProto(allow_soft_placement=True)
        config_fp = args.config_name
        init_checkpoint = args.ckpt_name
        logger.info('model config: %s' % config_fp)

        # 加载bert配置文件
        with tf.gfile.GFile(config_fp, 'r') as f: # edit by gavin: 类似于with open()
            bert_config = modeling.BertConfig.from_dict(json.load(f)) # bert_config类才能被BertModel调用

        logger.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

        # xla加速
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            # 获取所有要训练的变量
            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            # 共享卷积核
            with tf.variable_scope("pooling"):
                # 如果只有一层，就只取对应那一层的weight
                if len(args.layer_indexes) == 1:
                    encoder_layer = model.all_encoder_layers[args.layer_indexes[0]]
                else:
                    # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
                    all_layers = [model.all_encoder_layers[l] for l in args.layer_indexes]
                    encoder_layer = tf.concat(all_layers, -1)

                input_mask = tf.cast(input_mask, tf.float32)

                # 以下代码是句向量的生成方法，可以理解为做了一个卷积的操作，但是没有把结果相加, 卷积核是input_mask
                if pooling_strategy == PoolingStrategy.REDUCE_MEAN: # edit by gavin:类似于平均池化，每个字的向量之和/字的个数
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MAX: # edit by gavin: 类似于最大池化，取每个字的每个维度的最大值作为向量
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                        masked_reduce_max(encoder_layer, input_mask)], axis=1) # edit by gavin: 平均池化和最大池化concat
                elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                        pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1) # edit by gavin: 取CLS的向量作为句向量
                elif pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                        pooling_strategy == PoolingStrategy.SEP_TOKEN:
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, seq_len - 1], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes) # edit by gavin： 取最后的token作为句向量
                elif pooling_strategy == PoolingStrategy.NONE:
                    pooled = mul_mask(encoder_layer, input_mask) # edit by gavin：取整个矩阵作为句子的表征
                else:
                    raise NotImplementedError()

            pooled = tf.identity(pooled, 'final_encodes')

            output_tensors = [pooled]
            # edit by gavin: 保存为pb格式主要有两种方式
            # 第一种方式使用graph_util.convert_variables_to_constants()
            # 第二种方式使用tf.get_default_graph().as_graph_def()
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...')
            tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        #tmp_file = tempfile.NamedTemporaryFile('w', delete=True).name
        #r = random.randint(1, 1000)
        #tmp_file = "./tmp_graph"+str(r)
        tmp_file = "./tmp_graph11"
        logger.info('write graph to a tmp file: %s' % tmp_file)
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return tmp_file
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)