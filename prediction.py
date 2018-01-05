import os,sys

import mxnet as mx 
import numpy as np 
import codecs
sys.path.append('.')
from Iter_mutt import Test_Iter
import model

##### scp file #####
data_names = ["mfcc-static_dim-13", "mfcc-static_dim-14-speaker_index"]
data_name = data_names[0]
train_feats_scp  = "./scp/%s/feats/train.scp" % data_name 
dev_feats_scp    = "./scp/%s/feats/dev.scp" % data_name
eval_feats_scp   = "./scp/%s/feats/eval.scp" % data_name

train_labels_scp = "./scp/%s/labels/train.scp" % data_name
dev_labels_scp   = "./scp/%s/labels/dev.scp" % data_name

data_dim   = 13
label_dim  = 2
batch_size = 4
lr         = 0.00001
num_epoch  = 30
ndnn       = 4
dnnsize    = 2048
prefix     = "checkpoints/spoofing_mfcc_arc-%dx%d_lr-%f_batchsize-%d" % (ndnn, dnnsize, lr, batch_size)

data_names = ['data']
label_names = ['softmax_label']
data_shapes = [(batch_size, data_dim)]
label_shapes = [(batch_size, label_dim)]



#net = model.dnn(ndnn=ndnn, dnnsize=dnnsize)

#data_names = [x[0] for x in test_Iter.provide_data]
#print data_names;raw_input()
#mod = mx.mod.Module(symbol=net, context=mx.context.gpu(5), data_names=data_names,label_names=label_names)
model_prefix = prefix

symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 1)

sym = model.dnn(ndnn=ndnn, dnnsize=dnnsize)
mod = mx.mod.Module(symbol=sym, context=mx.context.gpu(5), data_names=data_names,label_names=label_names)
mod.bind(for_training=False, data_shapes=[('data', (4, 13))], label_shapes=[('softmax_label', (4, 2))])
mod.set_params(arg_params=arg_params, aux_params=aux_params)
result = []

fin = codecs.open(dev_feats_scp, 'r')
files = fin.readlines()
fin.close()

for F in files:
  print F
  test_Iter = Test_Iter(F, data_names, data_shapes, label_names, label_shapes, batch_size)
  evaluation = 0
  num_batch = 0
  for preds, i_batch, batch in  mod.iter_predict(test_Iter):
    num_batch += 1
    prediction = mod.get_outputs()
    #out = prediction[0].asnumpy().astype('float32')
    evaluation += prediction[0].asnumpy()[:, 0].astype('float32').sum(0)/batch_size
  evaluation = evaluation / num_batch
  result.append(evaluation)

fout = codecs.open("./result.txt", "w")
for i in range(len(result)):
  fout.write(str(result[i]) + '\n')
fout.close()


