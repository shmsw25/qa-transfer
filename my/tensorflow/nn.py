from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.util import nest
import tensorflow as tf

from my.tensorflow import flatten, reconstruct, add_wd, exp_mask


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_wd(wd)

    return out


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out


def softsel(target, logits, mask=None, scope=None):
    """

    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank-1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height))
            outs.append(out)
        concat_out = tf.concat(2, outs)
        return concat_out

def qr_layer(u_, m, d, gate_mask, gate_size, att_forget_bias, layer_idx, L, m_length, reg_layer, is_train=None, wd=0.0, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "qrn"):
        prev_u = tf.nn.dropout(u_, keep_prob)
        u_t = tf.tanh(linear([prev_u, m], d, True, wd=wd, scope=scope + 'u_t'))
        a = tf.cast(gate_mask, 'float') * tf.sigmoid(linear([prev_u, m], gate_size, True, wd=wd, scope=scope + 'a') - att_forget_bias)

        h = reg_layer(u_t, a, 1.0-a, scope = scope + 'h')
        if layer_idx+1 < L:
            rf, rb = tf.split(2, 2, tf.cast(gate_mask, 'float') \
                * tf.sigmoid(linear([prev_u, m], 2*gate_size, True, wd=wd, scope = scope + 'r')))
            u_t_rev = tf.reverse_sequence(u_t, m_length, 1)
            a_rev, rb_rev = tf.reverse_sequence(a, m_length, 1), tf.reverse_sequence(rb, m_length, 1)
            uf = reg_layer(u_t, a*rf, 1.0-a, scope=scope+'uf')
            ub_rev = reg_layer(u_t_rev, a_rev*rb_rev, 1.0-a_rev, scope='ub_rev')
            ub = tf.reverse_sequence(ub_rev, m_length, 1)
            u = uf + ub
        else:
            rf = rb = tf.zeros(a.get_shape().as_list())

        return (rf, rb, a, h, u)

class ReductionLayer(object):
    def __init__(self, batch_size, mem_size, hidden_size):
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.batch_size = batch_size
        N, M, d = batch_size, mem_size, hidden_size
        self.L = np.tril(np.ones([M, M], dtype='float32'))
        self.sL = np.tril(np.ones([M, M], dtype='float32'), k=-1)

    def __call__(self, u_t, a, b, scope=None):
        """

        :param u_t: [N, M, d]
        :param a: [N, M. 1]
        :param b: [N, M. 1]
        :param mask:  [N, M]
        :return:
        """
        N, M, d = self.batch_size, self.mem_size, self.hidden_size
        L, sL = self.L, self.sL
        with tf.name_scope(scope or self.__class__.__name__):
            L = tf.tile(tf.expand_dims(L, 0), [N, 1, 1])
            sL = tf.tile(tf.expand_dims(sL, 0), [N, 1, 1])
            logb = tf.log(b + 1e-9)
            logb = tf.concat(1, [tf.zeros([N, 1, 1]), tf.slice(logb, [0, 1, 0], [-1, -1, -1])])
            left = L * tf.exp(tf.batch_matmul(L, logb * sL))  # [N, M, M]
            right = a * u_t  # [N, M, d]
            u = tf.batch_matmul(left, right)  # [N, M, d]
        return u


class VectorReductionLayer(object):
    def __init__(self, batch_size, mem_size, hidden_size):
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.batch_size = batch_size
        N, M, d = batch_size, mem_size, hidden_size
        self.L = np.tril(np.ones([M, M], dtype='float32'))
        self.sL = np.tril(np.ones([M, M], dtype='float32'), k=-1)
        print (N, M, d)

    def __call__(self, u_t, a, b, scope=None):
        """

        :param u_t: [N, M, d]
        :param a: [N, M. d]
        :param b: [N, M. d]
        :param mask:  [N, M]
        :return:
        """
        N, M, d = self.batch_size, self.mem_size, self.hidden_size
        L, sL = self.L, self.sL
        with tf.name_scope(scope or self.__class__.__name__):
            L = tf.tile(tf.expand_dims(tf.expand_dims(L, 0), 0), [N, d, 1, 1])
            sL = tf.tile(tf.expand_dims(tf.expand_dims(sL, 0), 0), [N, d, 1, 1])
            logb = tf.log(b + 1e-9)  # [N, M, d]
            logb = tf.concat(1, [tf.zeros([N, 1, d]), tf.slice(logb, [0, 1, 0], [-1, -1, -1])])  # [N, M, d]
            logb = tf.expand_dims(tf.transpose(logb, [0, 2, 1]), -1)  # [N, d, M, 1]
            left = L * tf.exp(tf.batch_matmul(L, logb * sL))  # [N, d, M, M]
            right = a * u_t  # [N, M, d]
            right = tf.expand_dims(tf.transpose(right, [0, 2, 1]), -1)  # [N, d, M, 1]
            u = tf.batch_matmul(left, right)  # [N, d, M, 1]
            u = tf.transpose(tf.squeeze(u, [3]), [0, 2, 1])  # [N, M, d]
        return u


def qr_network(u, u_mask, m, x_mask, use_vector_gate, L, wd, keep_prob = 1.0, att_forget_bias = 2.5, scope='qrn'):
    # u : [N, d]
    # m : [N, M, d]
    N, M, d = tuple(m.get_shape().as_list())
    m_mask = tf.reduce_max(tf.cast(x_mask, 'int64'), 2, name='m_mask')
    gate_mask = tf.expand_dims(m_mask, -1)
    m_length = tf.reduce_sum(m_mask, 1, name='m_length')
    u_ = tf.tile(tf.expand_dims(u, 1), [1, M, 1])
    reg_layer = VectorReductionLayer(N, M, d) if use_vector_gate else ReductionLayer(N, M, d)
    gate_size = d if use_vector_gate else 1
    h = None
    rfs, rbs, as_, hs = [], [], [], []

    for layer_idx in range(L):
        with tf.name_scope("layer_{}".format(layer_idx)):
            h_, u_ = qr_layer(u_, m, d, gate_mask, gate_size, att_forget_bias, layer_idx, L, m_length, reg_layer, is_train=None, wd=wd, keep_prob=keep_prob, scope=scope) 
            rfs.append(rf)
            rbs.append(rb)
            as_.append(a)
            hs.append(h_)
            tf.get_variable_scope().reuse_variables()

    return rfs, rbs, as_, hs
