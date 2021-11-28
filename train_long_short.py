# long-term using attention

import pandas as pd
import time
import datetime
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.utils.data as Data
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import sys
import codecs

import preprocess_longshort as preprocess
import model_longshort as model

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


run_name = "user-aware-0.001-NYC"

log = open("log_" + run_name + ".txt", "w")
sys.stdout = log
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training Parameters
batch_size = 32
hidden_size = 128
num_layers = 1
num_epochs = 25
lr = 0.001

vocab_hour = 24
vocab_week = 7

embed_poi = 300
embed_cat = 100
embed_user = 50
embed_hour = 20
embed_week = 7

print("emb_poi :", embed_poi)
print("emb_user :", embed_user)
print("hidden_size :", hidden_size)
print("lr :", lr)

data = pd.DataFrame(pd.read_table("../input/dataset_TSMC2014_NYC.txt", header=None, encoding="latin-1"))
data.columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]

print("start preprocess")
#pre_data = preprocess.sliding_varlen(data, batch_size)
print("pre done")


with open("pre_data.txt", "rb") as f:
    pre_data = pickle.load(f)

with open("long_term.pk", "rb") as f:
    long_term = pickle.load(f)

with open("cat_candidate.pk", "rb") as f:
    cat_candi = pickle.load(f)

# with open('long_term_feature.pk','rb') as f:
# 	long_term_feature = pickle.load(f)
long_term_feature = [0]

cat_candi = torch.cat((torch.Tensor([0]), cat_candi))
cat_candi = cat_candi.long()

[vocab_poi, vocab_cat, vocab_user, len_train, len_test] = pre_data["size"]

loader_train = pre_data["loader_train"]
loader_test = pre_data["loader_test"]

print("train set size: ", len_train)
print("test set size: ", len_test)
print("vocab_poi: ", vocab_poi)
print("vocab_cat: ", vocab_cat)

print("Train the Model...")


Model = model.long_short(
    embed_user,
    embed_poi,
    embed_cat,
    embed_hour,
    embed_week,
    hidden_size,
    num_layers,
    vocab_poi + 1,
    vocab_cat + 1,
    vocab_user + 1,
    vocab_hour,
    long_term,
    cat_candi,
    # len(long_term_feature[0]),
)



userid_cursor = False
results_cursor = False

Model = Model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr)


def precision(indices, batch_y, k, count, delta_dist):
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            precision += 1
    return precision / count


def MAP(indices, batch_y, k, count):
    sum_precs = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        ranked_list = sort[:k]
        hists = 0
        for n in range(len(ranked_list)):
            if ranked_list[n].cpu().numpy() in batch_y[i].long().cpu().numpy():
                hists += 1
                sum_precs += hists / (n + 1)
    return sum_precs / count


def recall(indices, batch_y, k, count, delta_dist):
    recall_correct = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            recall_correct += 1
    return recall_correct / count


for epoch in range(num_epochs):
    Model = Model.train()
    total_loss = 0.0

    precision_1 = 0
    precision_5 = 0
    precision_10 = 0
    precision_20 = 0

    recall_5 = 0
    recall_10 = 0

    MAP_1 = 0
    MAP_5 = 0
    MAP_10 = 0
    MAP_20 = 0

    userid_wrong_train = {}
    userid_wrong_test = {}
    results_train = []
    results_test = []

    for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_train):
        Model.zero_grad()     
        users = batch_userid.cuda()
        hourids = Variable(hours.long()).cuda()

        batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (
            Variable(batch_x).cuda(),
            Variable(batch_x_cat).cuda(),
            Variable(batch_y).cuda(),
            Variable(hour_pre.long()).cuda(),
            Variable(week_pre.long()).cuda(),
        )

        poi_candidate = list(range(vocab_poi + 1))
        poi_candi = Variable(torch.LongTensor(poi_candidate)).cuda()
        cat_candi = Variable(cat_candi).cuda()
        outputs = Model(
            batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
        )  

        loss = 0
        for i in range(batch_x.size(0)):
            loss += loss_function(outputs[i, :, :], batch_y[i, :]).cuda()

        loss.backward()
        optimizer.step()

        total_loss += float(loss)

        outputs2 = outputs[:, -1, :]
        batch_y2 = batch_y[:, -1]

        out_p, indices = torch.sort(outputs2, dim=1, descending=True)
        count = float(len_train)
        delta_dist = 0
        precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
        precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
        precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
        precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

        MAP_1 += MAP(indices, batch_y2, 1, count)
        MAP_5 += MAP(indices, batch_y2, 5, count)
        MAP_10 += MAP(indices, batch_y2, 10, count)
        MAP_20 += MAP(indices, batch_y2, 20, count)

    print(
        "train:",
        "epoch: [{}/{}]\t".format(epoch, num_epochs),
        "loss: {:.4f}\t".format(total_loss),
        "precision@1: {:.4f}\t".format(precision_1),
        "precision@5: {:.4f}\t".format(precision_5),
        "precision@10: {:.4f}\t".format(precision_10),
        "precision@20: {:.4f}\t".format(precision_20),
        "MAP@1: {:.4f}\t".format(MAP_1),
        "MAP@5: {:.4f}\t".format(MAP_5),
        "MAP@10: {:.4f}\t".format(MAP_10),
        "MAP@20: {:.4f}\t".format(MAP_20),
    )

    savedir = "checkpoint_file/checkpoint_" + run_name  
    if not os.path.exists(savedir):    
        os.makedirs(savedir)
    savename = savedir + "/checkpoint" + "_" + str(epoch) + ".tar"  

    torch.save({"epoch": epoch + 1, "state_dict": Model.state_dict(),}, savename) 
    
    if epoch % 1 == 0:

        Model = Model.eval()

        total_loss = 0.0

        precision_1 = 0
        precision_5 = 0
        precision_10 = 0
        precision_20 = 0

        MAP_1 = 0
        MAP_5 = 0
        MAP_10 = 0
        MAP_20 = 0

        for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_test):
            Model.zero_grad()
            hourids = hours.long()
            users = batch_userid

            batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (
                Variable(batch_x).cuda(),
                Variable(batch_x_cat).cuda(),
                Variable(batch_y).cuda(),
                Variable(hour_pre.long()).cuda(),
                Variable(week_pre.long()).cuda(),
            )
            users = Variable(users).cuda()
            hourids = Variable(hourids).cuda()

            outputs = Model(
                batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
            ) 
            loss = 0
            for i in range(batch_x.size(0)):
                loss += loss_function(outputs[i, :, :], batch_y[i, :])

            total_loss += float(loss)

            outputs2 = outputs[:, -1, :]
            batch_y2 = batch_y[:, -1]

            weights_output = outputs2.data

            outputs2 = weights_output  # +weights_classify# + weights_comatrix +weights_hour_prob
            out_p, indices = torch.sort(outputs2, dim=1, descending=True)

            count = float(len_test)

            precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
            precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
            precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
            precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

            MAP_1 += MAP(indices, batch_y2, 1, count)
            MAP_5 += MAP(indices, batch_y2, 5, count)
            MAP_10 += MAP(indices, batch_y2, 10, count)
            MAP_20 += MAP(indices, batch_y2, 20, count)

        print(
            "val:",
            "loss: {:.4f}\t".format(total_loss),
            "precision@1: {:.4f}\t".format(precision_1),
            "precision@5: {:.4f}\t".format(precision_5),
            "precision@10: {:.4f}\t".format(precision_10),
            "precision@20: {:.4f}\t".format(precision_20),
            "MAP@1: {:.4f}\t".format(MAP_1),
            "MAP@5: {:.4f}\t".format(MAP_5),
            "MAP@10: {:.4f}\t".format(MAP_10),
            "MAP@20: {:.4f}\t".format(MAP_20),
        )


log.close()

