import random
import codecs
import mxnet as mx
import numpy as np
import data_loader

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None

    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    def provide_label(self):
        return [mx.io.DataDsec('softmax_label', self.label.shape)]


class Test_Iter(mx.io.DataIter):
    def __init__(self, feats_file, data_names, data_shapes, label_names, label_shapes, batch_size):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.cur_batch = 0
        self.feats_file = feats_file
        self.batch_size = batch_size
        self.features = data_loader.loader_by_frames(test_file=self.feats_file, do_shuffle=False)
	self.num_batches = len(self.features) / self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_batch = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch == self.num_batches:
	    raise StopIteration
        data = [mx.nd.array(self.features[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size])]
        label = [mx.nd.full((self.batch_size, 2), 3)]
        self.cur_batch += 1
        return mx.io.DataBatch(data, label, pad=0)


class Simple_Iter(mx.io.DataIter):
    def __init__(self, feats_scp, labels_scp, data_names, data_shapes, label_names, label_shapes, batch_size):
        self._provide_data  = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.cur_batch = 0
        self.feats_scp = feats_scp
        self.labels_scp = labels_scp
        self.batch_size = batch_size
	self.features, self.labels = data_loader.loader_by_frames(feats_scp=self.feats_scp, labels_scp=self.labels_scp)
        #self.features, self.labels = data_loader.load_cqcc(feat_scp=self.feats_scp, label_scp=self.labels_scp)
	self.scale = 50

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_batch = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
    	if len(self.features)/self.scale>((self.cur_batch + 1) * self.batch_size):
            data = [mx.nd.array(self.features[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size])]
            label = [mx.nd.array(self.labels[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size])]
	    self.cur_batch += 1
            return mx.io.DataBatch(data, label)

        elif len(self.features)/self.scale<(self.cur_batch * self.batch_size):
            raise StopIteration

	elif len(self.features)/self.scale<=((self.cur_batch + 1) * self.batch_size):
            residual = ((self.cur_batch + 1) * self.batch_size) - len(self.features)
            pad = 0
            data = [mx.nd.array(self.features[self.cur_batch * self.batch_size:] + [np.full((self.data_shapes[0][1],), pad)] * residual)]
            label = [mx.nd.array(self.labels[self.cur_batch * self.batch_size:] + [np.full((self.label_shapes[0][1],), pad)] * residual)]
	    self.cur_batch += 1
            return mx.io.DataBatch(data, label)

        else:
            print "something wrong happened in generate batches"
            exit(1)



class multi_input_Iter(mx.io.DataIter):

    def __init__(self, feats_scp, labels_scp, batch_size, feat_dim, label_dim, num_hidden, shuffle=True, mode="train"):

        self.mode           = mode
        self.num_hidden     = num_hidden
        self.data_name      = "data"
        self.label_name     = "softmax_label"
        self.sent_name      = "sentidx"
        self.feats_scp      = feats_scp
        self.labels_scp     = labels_scp

        self.feat_dim       = feat_dim
        self.label_dim      = label_dim


        self.batch_size     = batch_size
        self.shuffle        = shuffle

        self.data           = [mx.nd.zeros((self.batch_size, self.feat_dim-1))]
        self.data.append(mx.nd.zeros((self.batch_size, self.num_hidden)))
        for i in range(10):
            self.data.append(mx.nd.full((self.batch_size, self.num_hidden), i+1.0))
        self.label          = [mx.nd.zeros((self.batch_size, self.label_dim))]

        self.sents          = []
        for i in range(10):
            self.sents.append("sent%d" % (i+1))
        self.data_names     = [self.data_name, self.sent_name] + self.sents
        self.label_names    = [self.label_name]

        self._provide_data  = [mx.io.DataDesc(self.data_name, self.data[0].shape),
                               mx.io.DataDesc(self.sent_name, self.data[1].shape)]
        for i in range(10):
            self._provide_data.append(mx.io.DataDesc(self.sents[i], self.data[i+2].shape))

        self._provide_label = [(self.label_name, self.label[0].shape)]

        self.pack()
        self.reset()


    def __iter__(self):

        return self


    def pack(self):
        ''' packing data to batches '''

        fin = codecs.open(self.feats_scp);featfiles   = fin.readlines();fin.close()
        fin = codecs.open(self.labels_scp);labelfiles = fin.readlines();fin.close()
        assert(len(featfiles)==len(labelfiles))

        self.batches = []
        self.sent    = []
        self.pads    = []
        features     = []
        labels       = []
        sents     = []
        alldata         = []

        for f in range(len(featfiles)):


            temp_features = []
            temp_labels   = []
            temp_sents    = []
            temp_data     = []
            temp_batch    = []


            path = file(featfiles[f].strip().split('\n')[0], "rb")
            feat = np.load(path)
            path.close()
            basename_f = featfiles[f].strip().split('/')[-1].split('\n')[0].split('.')[0]
            print "process %s "% basename_f

            path  = file(labelfiles[f].strip().split('\n')[0], "rb")
            label = np.load(path)
            path.close()
            basename_l = labelfiles[f].strip().split('/')[-1].split('\n')[0].split('.')[0]
            print "process %s "% basename_l

            assert(basename_f == basename_l)

            nframe = feat.shape[0]
            for j in range(0, nframe):

                if self.mode == "train":

                    features.append(feat[j][:-1])
                    labels.append(label)
                    sents.append(feat[j][-1])
                    alldata.append((feat[j][:-1], label, feat[j][-1]))

                elif self.mode == "sent":

                    temp_features.append(feat[j][:-1])
                    temp_labels.append(label)
                    temp_sents.append(feat[j][-1])
                    temp_data.append((feat[j][:-1], label, feat[j][-1]))

                else:

                    print "Error: unknown mode!"
                    exit(1)

            if self.mode == "sent":

                temp_data  = np.array(temp_data)

                nbatch = len(temp_data) / self.batch_size
                residual = len(temp_data) - nbatch * self.batch_size

                for i in range(nbatch):
                    np_data_buffer = np.zeros((self.batch_size, self.feat_dim-1))
                    np_label_buffer = np.zeros((self.batch_size, self.label_dim))
                    np_sent_buffer = np.zeros((self.batch_size, self.num_hidden))
                    for x in range(self.batch_size):
                        np_data_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 0][x]
                        np_label_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 1][x]
                        np_sent_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 2][x]
                    self.data[0][:] = np_data_buffer
                    self.data[1][:] = np_sent_buffer
                    self.label[0][:] = np_label_buffer
                    tdata  = []
                    tlabel = []

                    for k in range(len(self.data)):
                        tdata.append(self.data[k].copy())
                    for k in range(len(self.label)):
                        tlabel.append(self.label[k].copy())
                    temp_batch.append((tdata, tlabel))

                if residual > 0:
                    residual_feats  = np.concatenate((temp_features[-residual:], np.zeros((self.batch_size - residual, self.feat_dim-1))))
                    residual_labels = np.concatenate((temp_labels[-residual:], np.zeros((self.batch_size - residual, self.label_dim))))
                    residual_sents  = np.concatenate((temp_sents[-residual:], np.ones((self.batch_size - residual))))
                    assert(residual_sents.shape[0]==residual_labels.shape[0])
                    residual_batch  = []

                    assert(residual_feats.shape[0]==residual_labels.shape[0])
                    np_data_buffer = np.zeros((self.batch_size, self.feat_dim-1))
                    np_label_buffer = np.zeros((self.batch_size, self.label_dim))
                    np_sent_buffer = np.zeros((self.batch_size, self.num_hidden))
                    for x in range(self.batch_size):
                        np_data_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 0][x]
                        np_label_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 1][x]
                        np_sent_buffer[x][:] = temp_data[i*self.batch_size:(i+1)*self.batch_size, 2][x]


                    self.data[0][:] = np_data_buffer
                    self.data[1][:] = np_sent_buffer
                    self.label[0][:] = np_label_buffer

                    tdata  = []
                    tlabel = []

                    for k in range(len(self.data)):
                        tdata.append(self.data[k].copy())
                    for k in range(len(self.label)):
                        tlabel.append(self.label[k].copy())
                    temp_batch.append((tdata, tlabel))


                self.sent.append(temp_batch)
                self.pads.append(residual)




        if self.mode == "train":

            if self.shuffle == True:

                random.shuffle(alldata)

            alldata    = np.array(alldata)
            batches     = []
            nbatch = len(alldata)/self.batch_size

            np_data_buffer = np.zeros((self.batch_size, self.feat_dim-1))
            np_label_buffer = np.zeros((self.batch_size, self.label_dim))
            np_sent_buffer = np.zeros((self.batch_size, self.num_hidden))

            for i in range(nbatch):
                for x in range(self.batch_size):
                    np_data_buffer[x][:] = alldata[i*self.batch_size:(i+1)*self.batch_size, 0][x]
                    np_label_buffer[x][:] = alldata[i*self.batch_size:(i+1)*self.batch_size, 1][x]
                    np_sent_buffer[x][:] = alldata[i*self.batch_size:(i+1)*self.batch_size, 2][x]
                self.data[0][:] = np_data_buffer
                self.data[1][:] = np_sent_buffer
                self.label[0][:] = np_label_buffer
                tdata  = []
                tlabel = []


                for k in range(len(self.data)):
                    tdata.append(self.data[k].copy())
                for k in range(len(self.label)):
                    tlabel.append(self.label[k].copy())
                self.batches.append((tdata, tlabel))


        self.num_batches = len(self.batches)
        print "finish packing"



    def reset(self):

        self.cur_batch = -1
        self.cur_sent  = -1


    def __next__(self):

        return self.next()


    @property
    def provide_data(self):

        return self._provide_data


    @property
    def provide_label(self):

        return self._provide_label


    def next(self):

        if self.mode == "sent" and self.cur_sent < (len(self.sent)-1):

            self.nsent = len(self.sent)

            batches_per_sent = []
        for b in range(len(self.sent[self.cur_sent])):
            mdata = self.sent[self.cur_sent][b][0]
            mlabel = self.sent[self.cur_sent][b][1]

            data_batch = SimpleBatch(self.data_names, mdata, self.label_names, mlabel)
            batches_per_sent.append(data_batch)

            self.cur_sent += 1

            return (batches_per_sent, self.pads[self.cur_sent])


        if self.mode == "train" and self.cur_batch < len(self.batches)-1:

            self.cur_batch += 1
            mdata = self.batches[self.cur_batch][0]
            mlabel = self.batches[self.cur_batch][1]

            data_batch = SimpleBatch(self.data_names, mdata, self.label_names, mlabel)

            return data_batch

        else:

            raise StopIteration

