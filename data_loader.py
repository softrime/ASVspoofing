import random
import numpy as np 
import codecs
import scipy.io as scio

def shuffle(features, labels):

  assert(len(features)==len(labels))
  num_frames = len(features)
  idx = [i for i in range(num_frames)]
  assert(len(idx)==len(features)==len(labels))
  random.shuffle(idx)
  shuffle_features = [features[idx[i]] for i in range(len(idx))]
  shuffle_labels   = [labels[idx[i]] for i in range(len(idx))]
  return shuffle_features, shuffle_labels

def loader_by_frames(test_file=None, feats_scp=None, labels_scp=None, do_shuffle=True):
  
  if feats_scp != None and labels_scp != None:
    fin = codecs.open(feats_scp)
    featfiles = fin.readlines()
    fin.close()

    fin = codecs.open(labels_scp)
    labelfiles = fin.readlines()
    fin.close()
    assert(len(featfiles)==len(labelfiles))
    features = []
    labels   = []

    for f in range(len(featfiles)):
      basename_feat = featfiles[f].strip().split('/')[-1].split('.')[0]
      print 'processing %s'%basename_feat
      path = file(featfiles[f].strip().split('\n')[0], 'rb')
      feat = np.load(path)
      path.close()

      basename_label = labelfiles[f].strip().split('/')[-1].split('.')[0]
      path  = file(labelfiles[f].strip().split('\n')[0], 'rb')
      label = np.load(path)
      path.close()
      assert(basename_feat==basename_label)

      num_frames = feat.shape[0]
      for j in range(num_frames):
        features.append(feat[j])
        labels.append(label)

    if do_shuffle==True:
      features, labels = shuffle(features, labels)
      
    return features, labels

  elif test_file != None:
    features = []
    path = file(test_file.strip().split('\n')[0], 'rb')
    feat = np.load(path)
    path.close()
    num_frames = feat.shape[0]
    for j in range(num_frames):
      features.append(feat[j])
    return features

def load_cqcc(test_file=None, feat_scp=None, label_scp=None, do_shuffle=True):
  if feat_scp != None and label_scp != None:
    fin = codecs.open(feat_scp, 'r')
    featfiles = fin.readlines()
    fin.close()
    fin = codecs.open(label_scp, 'r')
    labelfiles = fin.readlines()
    fin.close()
    assert(len(featfiles)==len(labelfiles))
    features = []
    labels = []

    for f in range(len(featfiles)):
      basename = featfiles[f].strip().split('/')[-1].split('.')[0]
      print "processing %s"% basename
      filename = 'data/cqcc/asvspoof/%s_cqcc.mat'% basename
      temp_data = scio.loadmat(filename)
      feat = temp_data['x']
      feat = feat.T
      #print feat, feat.shape;raw_input()

      path = file(labelfiles[f].strip().split('\n')[0], 'rb')
      label = np.load(path)
      path.close()
      for i in range(feat.shape[0]):
        features.append(feat[i])
	labels.append(label)

    assert(len(features)==len(labels))
    if do_shuffle==True:
      features, labels = shuffle(features, labels)
    return features, labels

  elif test_file != None:
    features = []
    basename = test_file.strip().split('/')[-1].split('.')[0]
    print "processing %s" % basename 
    filename = 'data/cqcc/asvspoof/%s_cqcc.mat'% basename
    temp_data = scio.loadmat(filename)
    feat = temp_data['x']
    feat = feat.T
    num_frames = feat.shape[0]
    for j in range(num_frames):
      features.append(feat[j])
    return features


if __name__=='__main__':
  load_cqcc('./scp/mfcc-static_dim-13/feats/train.scp')
