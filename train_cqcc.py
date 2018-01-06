
import os, sys
import time
import codecs
import random
import logging

import numpy as np
import mxnet as mx
from mxnet import nd

sys.path.append('.')
from Iter_cqcc import Simple_Iter
from Iter_cqcc import Test_Iter
import model
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
data_name = data_names[0]
train_feats_scp  = "./scp/%s/feats/train.scp" % data_name 
dev_feats_scp    = "./scp/%s/feats/dev.scp" % data_name
eval_feats_scp   = "./scp/%s/feats/eval.scp" % data_name

train_labels_scp = "./scp/%s/labels/train.scp" % data_name
dev_labels_scp   = "./scp/%s/labels/dev.scp" % data_name



##### parameters config #####
#data_dim   = 13
data_dim   = 90
label_dim  = 2
batch_size = 4
lr         = 0.000001
num_epoch  = 30
ndnn       = 50
dnnsize    = 1024
pfix       = "spoofing_cqcc_arc-%dx%d_lr-%f_batchsize-%d"% (ndnn, dnnsize, lr, batch_size)
prefix     = "checkpoints/spoofing_cqcc_arc-%dx%d_lr-%f_batchsize-%d" % (ndnn, dnnsize, lr, batch_size)


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
data_names = ['data']
label_names = ['softmax_label']
data_shapes = [(batch_size, data_dim)]
label_shapes = [(batch_size, label_dim)]
#train_Iter = Iter(train_feats_scp, train_labels_scp, batch_size, data_dim, label_dim, dnnsize, True, mode="train")
#dev_Iter   = Iter(dev_feats_scp, dev_labels_scp, batch_size, data_dim, label_dim, dnnsize, False, mode="train")
train_Iter = Simple_Iter(train_feats_scp, train_labels_scp, data_names, data_shapes, label_names, label_shapes, batch_size)
dev_Iter = Simple_Iter(dev_feats_scp, dev_labels_scp, data_names, data_shapes, label_names, label_shapes, batch_size)
print "finish iter"



##### define NN #####
net = model.dnn(ndnn=ndnn, dnnsize=dnnsize)

##### create and train #####
data_names = [x[0] for x in train_Iter.provide_data]
#print data_names;raw_input()
label_names = [x[0] for x in train_Iter.provide_label]
mod = mx.mod.Module(symbol=net, context=mx.context.gpu(5), data_names=data_names,label_names=label_names)
model_prefix = prefix
checkpoint = mx.callback.do_checkpoint(model_prefix)
print "finish model"


def dev_loss(module, eval_data, eval_metric):
    eval_data.reset()
    eval_metric.reset()
    for batch in eval_data:
        module.forward(batch, is_train=False)
	module.update_metric(eval_metric, batch.label)
	outputs = module.get_outputs()

'''
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
metric = mx.metric.create('rmse')
print "begin training"
mod.fit(train_data=train_Iter, 
              eval_data=dev_Iter,
              optimizer='sgd', 
              optimizer_params={'learning_rate':lr}, 
              eval_metric=metric, 
              num_epoch=num_epoch,
              epoch_end_callback=checkpoint)  



##### predict in dev set  #####
print 'predict in dev set'
fin = codecs.open(dev_feats_scp, 'r')
files = fin.readlines()
fin.close()

fout = codecs.open("./result/dnn/%s.dev.result"% pfix, "w")
for F in files:
    basename = F.strip().split('/')[-1].split('.')[0]
    test_Iter = Test_Iter(F, data_names, data_shapes, label_names, label_shapes, batch_size)
    evaluation = 0
    num_batch = 0
    for preds, i_batch, batch in  mod.iter_predict(test_Iter):
        num_batch += 1
	prediction = mod.get_outputs()
        evaluation += prediction[0].asnumpy()[:, 0].astype('float32').sum(0)/batch_size
    evaluation = evaluation / num_batch
    fout.write(basename + ' ' + str(evaluation) + '\n')
fout.close()


##### predict in eval set #####
print 'predict in eval set'
fin = codecs.open(eval_feats_scp, 'r')
files = fin.readlines()
fin.close()

fout = codecs.open("./result/dnn/%s.eval.result"% pfix, "w")
for F in files:
    basename = F.strip().split('/')[-1].split('.')[0]
    test_Iter = Test_Iter(F, data_names, data_shapes, label_names, label_shapes, batch_size)
    evaluation = 0 
    num_batch = 0
    for preds, i_batch, batch in  mod.iter_predict(test_Iter):
        num_batch += 1
        prediction = mod.get_outputs()
        evaluation += prediction[0].asnumpy()[:, 0].astype('float32').sum(0)/batch_size 
    evaluation = evaluation / num_batch
    fout.write(basename + ' ' + str(evaluation) + '\n')
fout.close()

'''
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

