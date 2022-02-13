
import torch.nn as nn
import torch
from torch.autograd import Variable  
from torch.nn.parameter import Parameter
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
import numpy as np

class long_short(nn.Module):
	def __init__(self,embed_size_user, embed_size_poi, embed_size_cat,embed_size_hour, embed_size_week,
		hidden_size, num_layers, vocab_poi, vocab_cat,vocab_user,vocab_hour,long_term,cat_candi):

		super(long_short, self).__init__()
		
		self.embed_user = nn.Embedding(vocab_user, embed_size_user)
		self.embed_poi = nn.Embedding(vocab_poi, embed_size_poi)
		self.embed_cat = nn.Embedding(vocab_cat, embed_size_cat)
		self.embed_hour = nn.Embedding(vocab_hour, embed_size_hour)
		self.embed_week = nn.Embedding(7, embed_size_week)
		
		self.embed_total_size = embed_size_poi + embed_size_cat + embed_size_hour + embed_size_week 
		self.vocab_poi = vocab_poi
		self.vocab_hour = vocab_hour
		self.vocab_week = 7
		self.long_term = long_term


		self.weight_poi = Parameter(torch.ones(embed_size_poi,embed_size_user))
		self.weight_cat = Parameter(torch.ones(embed_size_cat,embed_size_user))
		self.weight_time = Parameter(torch.ones(embed_size_hour + embed_size_week,embed_size_user))
		self.bias = Parameter(torch.ones(embed_size_user))
		
		self.activate_func = nn.ReLU()

		self.num_layers = num_layers
		self.hidden_size = hidden_size
		
		self.out_w_long = Parameter(torch.Tensor([0.5]).repeat(vocab_user))
		self.out_w_poi = Parameter(torch.Tensor([0.25]).repeat(vocab_user))
		self.out_w_cat = Parameter(torch.Tensor([0.25]).repeat(vocab_user))

		
		#self.w1 = Parameter(torch.Tensor([0.5]))
		#self.w2 = Parameter(torch.Tensor([0.5]))
		#self.w3 = Parameter(torch.Tensor([0.5]))

		self.weight_hidden_poi = Parameter(torch.ones(self.hidden_size,1))
		self.weight_hidden_cat = Parameter(torch.ones(self.hidden_size,1))

		self.vocab_poi = vocab_poi
		self.vocab_cat = vocab_cat
		size = embed_size_poi + embed_size_user + embed_size_hour
		size2 = embed_size_cat + embed_size_user + embed_size_hour

		self.lstm_poi = nn.LSTM(size, hidden_size, num_layers, dropout = 0.5, batch_first = True)
		self.lstm_cat = nn.LSTM(size2, hidden_size, num_layers, dropout = 0.5, batch_first = True)
		self.fc_poi = nn.Linear(hidden_size,self.vocab_poi)
		self.fc_cat = nn.Linear(hidden_size,self.vocab_poi)
		self.attn_linear = nn.Linear(self.hidden_size*2, self.vocab_poi)

		self.fc_longterm = nn.Linear(self.embed_total_size, self.vocab_poi)

	def forward(self, inputs_poi, inputs_cat, inputs_user,inputs_time,hour_pre,week_pre,poi_candi,cat_candi):#,inputs_features):


		out_poi = self.get_output(inputs_poi,inputs_user,inputs_time,self.embed_poi,self.embed_user,self.embed_hour,self.lstm_poi,self.fc_poi)
		out_cat = self.get_output(inputs_cat,inputs_user,inputs_time,self.embed_cat,self.embed_user,self.embed_hour,self.lstm_cat,self.fc_cat)


#########################################################################################

		# long-term preference 
		
		u_long = self.get_u_long(inputs_user)

		# with fc_layer
		out_long = self.fc_longterm(u_long).unsqueeze(1).repeat(1,out_poi.size(1),1)

#########################################################################################
	#output 

	# weighted sum directly
		weight_poi = self.out_w_poi[inputs_user]
		weight_cat = self.out_w_cat[inputs_user]
		#weight_long = self.out_w_long[inputs_user] #32
		weight_long = 1-weight_poi-weight_cat

	
		out_poi = torch.mul(out_poi, weight_poi.unsqueeze(1).repeat(1,19).unsqueeze(2))
		out_cat = torch.mul(out_cat, weight_cat.unsqueeze(1).repeat(1,19).unsqueeze(2))
		out_long = torch.mul(out_long, weight_long.unsqueeze(1).repeat(1,19).unsqueeze(2))
		
		out = out_poi + out_cat + out_long

		return out

	def get_u_long(self,inputs_user):
		# get the hidden vector of users' long-term preference 

		u_long = {}
		for user in inputs_user:

			user_index = user.tolist()
			if user_index not in u_long.keys():

				poi = self.long_term[user_index]['loc']
				hour = self.long_term[user_index]['hour']
				week = self.long_term[user_index]['week']
				cat = self.long_term[user_index]['category']

				seq_poi = self.embed_poi(poi)
				seq_cat = self.embed_cat(cat)
				seq_user = self.embed_user(user)
				seq_hour = self.embed_hour(hour)
				seq_week = self.embed_week(week)
				seq_time = torch.cat((seq_hour, seq_week),1)

				poi_mm = torch.mm(seq_poi, self.weight_poi)
				cat_mm = torch.mm(seq_cat, self.weight_cat)
				time_mm = torch.mm(seq_time, self.weight_time)

				hidden_vec =  poi_mm.add_(cat_mm).add_(time_mm).add_(self.bias)
				hidden_vec = self.activate_func(hidden_vec)# 876*50
				alpha = F.softmax(torch.mm(hidden_vec, seq_user.unsqueeze(1)),0) #876*1

				poi_concat = torch.cat( (seq_poi,seq_cat, seq_hour, seq_week), 1) #876*427

				u_long[user_index] = torch.sum( torch.mul(poi_concat, alpha),0 )

		
		u_long_all = torch.zeros(len(inputs_user),self.embed_total_size).cuda()
		#64*427
		for i in range(len(inputs_user)):
			u_long_all[i,:] = u_long[inputs_user.tolist()[i]]
		
		return u_long_all



	def get_output(self, inputs,inputs_user,inputs_time,embed_layer,embed_user,embed_time,lstm_layer,fc_layer):

			# embed your sequences
		seq_tensor = embed_layer(inputs)
		seq_user = embed_user(inputs_user).unsqueeze(1).repeat(1,seq_tensor.size(1),1)
		seq_time = embed_time(inputs_time)
			# embed your sequences
		input_tensor = torch.cat((seq_tensor,seq_user,seq_time),2)
			# pack them up nicely
		output, _ = lstm_layer(input_tensor)
		out = fc_layer(output) # the last outputs
		return out


