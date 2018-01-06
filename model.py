import mxnet as mx

def dnn_multi_input(ndnn=1, dnnsize=64):
  sentidx = mx.sym.Variable("sentidx")
  sents = []
  for i in range(0, 10):
    sents.append(mx.sym.Variable('sent%d'% (i+1)))
  weight = []
  bias   = []

  data = mx.sym.Variable('data')
  next_layer = []
  for i in range(0, 10):
    net = mx.sym.FullyConnected(data=data, name="fc1_%d"%(i+1), num_hidden=dnnsize)
    #bn = mx.symbol.BatchNorm(data=net, name="bn1_%d"%(i+1), eps=1e-5, fix_gamma=False, use_global_stats=False)
    bn = mx.symbol.BatchNorm(data=net)
    net = mx.sym.Activation(data=bn, name="sig1_%d"%(i+1), act_type="sigmoid")
    next_layer.append(net)

  select = next_layer[0]
  for i in range(0, 10):
    select = mx.sym.where(mx.sym.broadcast_equal(sentidx, sents[i]), next_layer[i], select)


  for l in range(0, ndnn):
    net = mx.sym.FullyConnected(data=select, name="l%d" % (l+2), num_hidden=dnnsize)
    #bn = mx.symbol.BatchNorm(data=net, name="bn%d"%(i+2), eps=1e-5, fix_gamma=False, use_global_stats=False)
    bn = mx.symbol.BatchNorm(data=net)
    net = mx.sym.Activation(data=bn, name="sig%d" % (l+2), act_type="sigmoid")
  net = mx.sym.FullyConnected(data=net, name="fc%d" % (ndnn + 2), num_hidden=2)
  net = mx.sym.SoftmaxOutput(data=net, name='softmax')
  return net


def dnn(ndnn=3, dnnsize=1024):
  net = mx.sym.Variable('data')
  for i in range(ndnn):
    net = mx.sym.FullyConnected(data=net, name='fc%d'%(i+1), num_hidden=dnnsize)
    net = mx.symbol.BatchNorm(data=net)
    net = mx.sym.Activation(data=net, name='sig%d'%(i+1), act_type='relu')
    net = mx.sym.Dropout(data=net, p=0.5, name='dp%d'%(i+1))
  net = mx.sym.FullyConnected(data=net, name='fc%d'%(ndnn+1), num_hidden=2)
  net = mx.sym.SoftmaxOutput(data=net, name='softmax')
  return net


def cnn():
  net = mx.sym.Variable('data')
  conv1 = mx.symbol.Convolution(data=net, kernel=(5,5), num_filter=20)
  bn1   = mx.symbol.BatchNorm(data=conv1)
  tanh1 = mx.symbol.Activation(data=bn1, act_type="tanh")
  pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                            kernel=(2,2), stride=(2,2))

  conv2 = mx.symbol.Convolution(data=pool1, kernel=(4,4), num_filter=50)
  bn2   = mx.symbol.BatchNorm(data=conv2)  
  tanh2 = mx.symbol.Activation(data=bn2, act_type="tanh")
  pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                            kernel=(2,2), stride=(2,2))

  flatten = mx.symbol.Flatten(data=pool2)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  bn3 = mx.symbol.BatchNorm(data=fc1)
  tanh4 = mx.symbol.Activation(data=bn3, act_type="tanh")
  dp = mx.sym.Dropout(data=tanh4, p=0.5) 

  fc2 = mx.symbol.FullyConnected(data=dp, num_hidden=2)
  lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
  return lenet

