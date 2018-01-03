
import os, sys
import time
import codecs
import random
import logging

import numpy as np
import mxnet as mx
from mxnet import nd

sys.path.append('.')
sys.path.append('./nn')
from Iter_mutt import Iter
from dnn_multi_input import dnn_multi_input

##### metric define #####
def MSE(labels, preds):
        labels = labels.reshape((-1,))
        preds = preds.reshape((-1,))
        loss = 0.
        num_inst = 0
        loss = ((labels - preds) ** 2).sum(0)
        #if loss < 1:
        #    print labels, preds, loss;raw_input('why so small mse')
        num_inst = labels.shape[0]
        return loss


##### scp file #####
data_names = ["mfcc-static_dim-13", "mfcc-static_dim-14-speaker_index"]
data_name = data_names[1]
train_feats_scp  = "./scp/%s/feats/train.scp" % data_name 
dev_feats_scp    = "./scp/%s/feats/dev.scp" % data_name
eval_feats_scp   = "./scp/%s/feats/eval.scp" % data_name

train_labels_scp = "./scp/%s/labels/train.scp" % data_name
dev_labels_scp   = "./scp/%s/labels/dev.scp" % data_name



##### parameters config #####
data_dim   = 14
label_dim  = 2
batch_size = 4
lr         = 0.000001
num_epoch  = 90
ndnn       = 4
dnnsize    = 2048
prefix     = "checkpoints/spoofing_mfcc_arc-%dx%d_lr-%f_batchsize-%d" % (ndnn, dnnsize, lr, batch_size)


##### log deploy #####
logfn      = "LOG/spoofing_mfcc_arc-%dx%d_lr-%f_batchsize-%d_epoch-%d_datad2.log" % (ndnn, dnnsize, lr, batch_size, num_epoch)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=logfn,
                    filemode='w')
console    = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')


##### create dataIter #####
train_Iter = Iter(train_feats_scp, train_labels_scp, batch_size, data_dim, label_dim, dnnsize, True, mode="train")
dev_Iter   = Iter(dev_feats_scp, dev_labels_scp, batch_size, data_dim, label_dim, dnnsize, False, mode="train")
print "finish iter"



##### define NN #####
'''
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
    net = mx.sym.Activation(data=net, name="sig1_%d"%(i+1), act_type="sigmoid")
    next_layer.append(net)

select = next_layer[0]
for i in range(0, 10):
    select = mx.sym.where(mx.sym.broadcast_equal(sentidx, sents[i]), next_layer[i], select)

 
for l in range(0, ndnn):
    net = mx.sym.FullyConnected(data=select, name="l%d" % (l+2), num_hidden=dnnsize)
    net = mx.sym.Activation(data=net, name="sig%d" % (l+2), act_type="sigmoid")
net = mx.sym.FullyConnected(data=net, name="fc%d" % (ndnn + 2), num_hidden=2)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
'''
net = dnn_multi_input(ndnn=ndnn, dnnsize=dnnsize)


##### create and train #####
data_names = [x[0] for x in train_Iter.provide_data]
#print data_names;raw_input()
label_names = [x[0] for x in train_Iter.provide_label]
mod = mx.mod.Module(symbol=net, context=mx.context.gpu(5), data_names=data_names,label_names=label_names)
model_prefix = prefix
checkpoint = mx.callback.do_checkpoint(model_prefix)



def dev_loss(module, eval_data, eval_metric):
    eval_data.reset()
    eval_metric.reset()
    for batch in eval_data:
        module.forward(batch, is_train=False)
	module.update_metric(eval_metric, batch.label)
	outputs = module.get_outputs()


mod.bind(data_shapes=train_Iter.provide_data, label_shapes=train_Iter.provide_label)
mod.init_params(initializer=mx.init.Uniform(scale=.1))
metric = mx.metric.create(MSE)
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', lr), ))

metric.reset()
dev_loss(mod, train_Iter, metric)
for name, val in metric.get_name_value():
    logging.info('Initialization Train-%s=%f', name, val)
metric.reset()
dev_loss(mod, dev_Iter, metric)
for name, val in metric.get_name_value():
    logging.info('Initialization Dev-%s=%f', name, val)
epoch = 0
last_acc = float("Inf")
last_params = None
while True:
    print "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh new epoch hhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
    tic = time.time()
    train_Iter.reset()
    metric.reset()
    for batch in train_Iter:
    	mod.forward(batch, is_train=True)
	mod.update_metric(metric, batch.label)
	mod.backward()
	mod.update()
    for name, val in metric.get_name_value():
        logging.info('Epoch[%d] Train-%s=%f', epoch, name, val)
    toc = time.time()
    logging.info('Epoch[%d] Time cost=%.3f', epoch, toc-tic)

    metric.reset()
    dev_loss(mod, dev_Iter, metric)
    curr_acc = None
    for name, val in metric.get_name_value():
        logging.info("Epoch[%d] Dev-%s=%f", epoch, name, val)
	curr_acc = val
    if epoch > 0 and curr_acc > last_acc:
        logging.info('Epoch[%d]: Dev set performance drops, reverting this epoch', epoch)
	logging.info('Epoch[%d]: LR decay: %g => %g', epoch, lr, lr/2.)
	lr = lr/2.
	mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', lr), ), force_init=True)	
	mod.set_params(*last_params)
    else:
    	last_params = mod.get_params()
	last_acc = curr_acc
	epoch += 1
	mx.model.save_checkpoint(model_prefix, epoch, mod.symbol, *last_params)
    if epoch == num_epoch :
    	break
		
	




'''
mod.fit(train_data=train_Iter, 
              eval_data=dev_Iter,
              optimizer='sgd', 
              optimizer_params={'learning_rate':lr}, 
              eval_metric=metric, 
              num_epoch=num_epoch,
              epoch_end_callback=checkpoint)  
'''


##### predict in dev set  #####
sent_Iter = Iter(dev_feats_scp, dev_labels_scp, batch_size, data_dim, label_dim, dnnsize, False, mode="sent")
sent_Iter.reset()
print "finish sentIter"


conf = []
prediction = []
for sent, sent_pad in sent_Iter:
    sent_pred = []
    #print "pad:", sent_pad
    label = sent[0].label
    for idx in range(len(sent)):
        batch = sent[idx]
        mod.forward(batch, is_train=False)
        outputs = [out[0:out.shape[0]-sent_pad] for out in mod.get_outputs()]

        #print outputs;raw_input('outputs')

        sent_pred.append(outputs[0])

    true  = []
    spoof = []
    for i in range(len(sent_pred)):
             
        for j in range((sent_pred[i].shape[0])):
            true.append(sent_pred[i][j][0])
            spoof.append(sent_pred[i][j][1])

    temp = 0.
    for i in range(len(true)):
        temp += true[i]
    #print temp,len(true);raw_input('true')
    true_conf = temp/len(true)
    
    temp = 0.
    for i in range(len(spoof)):  
        temp += spoof[i]
    spoof_conf = temp/len(spoof)
    #print temp, len(spoof);raw_input('spoof')
    true_conf = true_conf.asnumpy()
    prediction.append(true_conf)
    spoof_conf = spoof_conf.asnumpy()

    true  = label[0][0][0]
    spoof = label[0][0][1]
    true = true.asnumpy()
    spoof = spoof.asnumpy()
    conf.append(true_conf)
    #print true_conf, true;
    #if true_conf > spoof_conf:
    #    prediction.append((1, true))
    #else :
    #    prediction.append((0, true))

resultfn = "result/spoofing_mfcc_arc-%dx%d_lr-%f_batchsize-%d.result" % (ndnn, dnnsize, lr, batch_size) 
fout = codecs.open(resultfn, 'w')
dev_scp = "dev.scp"
fin = codecs.open(dev_scp, 'r');fns = fin.readlines();fin.close()
assert(len(fns)==len(prediction))
for i in range(len(fns)):
    fout.write(fns[i].split('.')[0] + ' ' + str(prediction[i]) + '\n')

'''
correct = 0.
for s in range(len(prediction)):
    fout.write(str(prediction[s][0]) + ' ' + str(prediction[s][1]) + '\n')
    if prediction[s][0]==prediction[s][1]:
        correct +=1

print correct, len(prediction)
accuracy = correct/len(prediction)
print accuracy

true_conf_mean = np.sum(conf[0:len(conf)/2])/(len(conf)/2)
spoof_conf_mean = np.sum(conf[len(conf)/2:])/(len(conf)/2)

print "mean:", true_conf_mean, spoof_conf_mean

fout.write("\nresult:\n" + "correct: " + str(correct) + "\nall: " + str(len(prediction)) + "\naccuracy: " + str(accuracy))
fout.write("\nmean:\n" + "true:" + str(true_conf_mean) + '\nspoof:' + str(spoof_conf_mean))
fout.close()
'''

