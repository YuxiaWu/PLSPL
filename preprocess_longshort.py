import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F	   
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle 
import time
import os

def sliding_varlen(data,batch_size):
	def timedelta(time1,time2):
		t1 = datetime.datetime.strptime(str(time1),'%a %b %d %H:%M:%S %z %Y')
		t2 = datetime.datetime.strptime(str(time2),'%a %b %d %H:%M:%S %z %Y')
		delta = t1-t2
		time_delta = datetime.timedelta(days = delta.days,seconds = delta.seconds).total_seconds()
		return time_delta/3600
	
	def get_entropy(x):
		x_value_list = set([x[i] for i in range(x.shape[0])])
		ent = 0.0
		for x_value in x_value_list:
			p = float(x[x == x_value].shape[0]) / x.shape[0]
			logp = np.log2(p)
			ent -= p * logp
		return ent

#################################################################################
	# 1、sort the raw data in chronological order
	timestamp = []
	hour = []
	day = []
	week = []
	hour_48 = []
	for i in range(len(data)):
		times = data['time'].values[i]
		timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y')))
		t = datetime.datetime.strptime(times,'%a %b %d %H:%M:%S %z %Y')
		year = int(t.strftime('%Y'))
		day_i = int(t.strftime('%j'))
		week_i = int(t.strftime('%w'))
		hour_i = int(t.strftime('%H'))
		hour_i_48 = hour_i
		if week_i == 0 or week_i == 6:
			hour_i_48 = hour_i + 24

		if year == 2013:
			day_i = day_i + 366
		day.append(day_i)
		
		hour.append(hour_i)
		hour_48.append(int(hour_i_48))
		week.append(week_i)

	data['timestamp'] = timestamp
	data['hour'] = hour
	data['day'] = day
	data['week'] = week
	data['hour_48'] = hour_48

	data.sort_values(by = 'timestamp',inplace=True,ascending = True)

#################################################################################
	# 2、filter users and POIs 
	
	'''
	thr_venue = 1
	thr_user = 20
	user_venue = data.loc[:,['userid','venueid']]
	#user_venue = user_venue.drop_duplicates()
	
	venue_count = user_venue['venueid'].value_counts()
	venue = venue_count[venue_count.values>thr_venue]
	venue_index =  venue.index
	data = data[data['venueid'].isin(venue_index)]
	user_venue = user_venue[user_venue['venueid'].isin(venue_index)]
	del venue_count,venue,venue_index
	
	#user_venue = user_venue.drop_duplicates()
	user_count = user_venue['userid'].value_counts()
	user = user_count[user_count.values>thr_user]
	user_index = user.index
	data = data[data['userid'].isin(user_index)]
	user_venue = user_venue[user_venue['userid'].isin(user_index)]
	del user_count,user,user_index
	
	user_venue = user_venue.drop_duplicates()
	user_count = user_venue['userid'].value_counts()
	user = user_count[user_count.values>1]
	user_index = user.index
	data = data[data['userid'].isin(user_index)]
	del user_count,user,user_index
	
	'''
	data['userid'] = data['userid'].rank(method='dense').values
	data['userid'] = data['userid'].astype(int)
	data['venueid'] =data['venueid'].rank(method='dense').values
	data['userid'] = data['userid'].astype(int)
	for venueid,group in data.groupby('venueid'):
		indexs = group.index
		if len(set(group['catid'].values))>1:
			for i in range(len(group)):
				data.loc[indexs[i],'catid'] = group.loc[indexs[0]]['catid']
	
	data = data.drop_duplicates()
	data['catid'] =data['catid'].rank(method='dense').values
	
#################################################################################
	poi_cat = data[['venueid','catid']]
	poi_cat = poi_cat.drop_duplicates()
	poi_cat = poi_cat.sort_values(by = 'venueid')
	cat_candidate = torch.Tensor(poi_cat['catid'].values)

	with open('cat_candidate.pk','wb') as f:
		pickle.dump(cat_candidate,f)

	# 3、split data into train set and test set.
	#    extract features of each session for classification

	vocab_size_poi = int(max(data['venueid'].values))
	vocab_size_cat = int(max(data['catid'].values))
	vocab_size_user = int(max(data['userid'].values))

	print('vocab_size_poi: ',vocab_size_poi)
	print('vocab_size_cat: ',vocab_size_cat)
	print('vocab_size_user: ',vocab_size_user)

	train_x  = []
	train_x_cat  = []
	train_y = []
	train_hour = []
	train_userid = []
	train_indexs = []

# the hour and week to be predicted
	train_hour_pre = []
	train_week_pre = []


	test_x  = []
	test_x_cat  = []
	test_y = []
	test_hour = []
	test_userid = []
	test_indexs = []

# the hour and week to be predicted
	test_hour_pre = []
	test_week_pre = []


	long_term = {}

	long_term_feature = []

	data_train = {}
	train_idx = {}
	data_test = {}
	test_idx = {}

	data_train['datainfo'] = {'size_poi':vocab_size_poi+1,'size_cat':vocab_size_cat+1,'size_user':vocab_size_user+1} 
	
	len_session = 20
	user_lastid = {}
#################################################################################
	# split data

	for uid, group in data.groupby('userid'):
		data_train[uid] = {}
		data_test[uid] = {}
		user_lastid[uid] = []
		inds_u = group.index.values
		split_ind = int(np.floor(0.8*len(inds_u)))
		train_inds = inds_u[:split_ind]
		test_inds = inds_u[split_ind:]

	#get the features of POIs for user uid
		#long_term_feature.append(get_features(group.loc[train_inds]))

		long_term[uid] = {}
		'''
		long_term[uid]['loc'] = []
		long_term[uid]['hour'] = []
		long_term[uid]['week'] = []
		long_term[uid]['category'] = []
	
		lt_data = group.loc[train_inds]
		long_term[uid]['loc'].append(lt_data['venueid'].values)
		long_term[uid]['hour'].append(lt_data['hour'].values)
		long_term[uid]['week'].append(lt_data['week'].values)
		long_term[uid]['category'].append(lt_data['catid'].values)
		'''
		lt_data = group.loc[train_inds]
		long_term[uid]['loc'] = torch.LongTensor(lt_data['venueid'].values).cuda()
		long_term[uid]['hour'] = torch.LongTensor(lt_data['hour'].values).cuda()
		long_term[uid]['week'] = torch.LongTensor(lt_data['week'].values).cuda()
		long_term[uid]['category'] = torch.LongTensor(lt_data['catid'].values).cuda()
	#split the long sessions to some short ones with len_session = 20

		train_inds_split = []
		num_session_train =int(len(train_inds)//(len_session))
		for i in range(num_session_train):
			train_inds_split.append(train_inds[i*len_session:(i+1)*len_session])
		if num_session_train<len(train_inds)/len_session:
			train_inds_split.append(train_inds[-len_session:])
		
		train_id = list(range(len(train_inds_split))) 

		test_inds_split = []
		num_session_test = int(len(test_inds)//(len_session))
		for i in range(num_session_test):
			test_inds_split.append(test_inds[i*len_session:(i+1)*len_session])
		if num_session_test<len(test_inds)/len_session:
			test_inds_split.append(test_inds[-len_session:])
		
		test_id = list(range(len(test_inds_split)+len(train_inds_split)))[-len(test_inds_split):]

		train_idx[uid] = train_id[1:]
		test_idx[uid] = test_id

		for ind in train_id:

		#generate data for comparative methods such as deepmove

			if ind == 0:
				continue

			data_train[uid][ind] = {}
			history_ind =[]
			for i in range(ind):
				history_ind.extend(train_inds_split[i])
			whole_ind = []
			whole_ind.extend(history_ind)
			whole_ind.extend(train_inds_split[ind])

			whole_data = group.loc[whole_ind]

			loc = whole_data['venueid'].values[:-1]
			tim = whole_data['hour'].values[:-1]
			target = group.loc[train_inds_split[ind][1:]]['venueid'].values

			#loc = group_i['venueid'].values[:-1]
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_train[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1)
			data_train[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1)
			data_train[uid][ind]['target'] = torch.LongTensor(target)

			user_lastid[uid].append(loc[-1])

			group_i = group.loc[train_inds_split[ind]]
		
			#generate data for SHAN
			current_loc = group_i['venueid'].values
			data_train[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1)
			#group_i = whole_data
		#generate data for my methods. X,Y,time,userid
			
			current_cat = group_i['catid'].values
			train_x.append(current_loc[:-1])
			train_x_cat.append(current_cat[:-1])
			train_y.append(current_loc[1:])
			#train_hour.append(group_i['hour_48'].values[:-1])
			train_hour.append(group_i['hour'].values[:-1])
			train_userid.append(uid)
			
			#train_hour_pre.append(group_i['hour'].values[-1])
			#train_week_pre.append(group_i['week'].values[-1])

			train_hour_pre.append(group_i['hour'].values[1:])
			train_week_pre.append(group_i['week'].values[1:])

			train_indexs.append(group_i.index.values)

		for ind in test_id:

			data_test[uid][ind] = {}
			history_ind =[]
			for i in range(len(train_inds_split)):
				history_ind.extend(train_inds_split[i])
			whole_ind = []
			whole_ind.extend(history_ind)
			whole_ind.extend(test_inds_split[ind-len(train_inds_split)])

			whole_data = group.loc[whole_ind]

			loc = whole_data['venueid'].values[:-1]
			tim = whole_data['hour'].values[:-1]
			target = group.loc[test_inds_split[ind-len(train_inds_split)][1:]]['venueid'].values

			#loc = group_i['venueid'].values[:-1]
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_test[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1)
			data_test[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1)
			data_test[uid][ind]['target'] = torch.LongTensor(target)

			user_lastid[uid].append(loc[-1])

			#group_i = whole_data

			group_i = group.loc[test_inds_split[ind-len(train_inds_split)]]

			current_loc = group_i['venueid'].values
			data_test[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1)

			current_cat = group_i['catid'].values
			test_x_cat.append(current_cat[:-1])
			test_x.append(current_loc[:-1])
			test_y.append(current_loc[1:])
			#test_hour.append(group_i['hour_48'].values[:-1])
			test_hour.append(group_i['hour'].values[:-1])
			test_userid.append(uid)

			#test_hour_pre.append(group_i['hour'].values[-1])
			#test_week_pre.append(group_i['week'].values[-1])

			test_hour_pre.append(group_i['hour'].values[1:])
			test_week_pre.append(group_i['week'].values[1:])

			test_indexs.append(group_i.index.values)


	with open('data_train.pk','wb') as f:
		pickle.dump(data_train,f)

	with open('data_test.pk','wb') as f:
		pickle.dump(data_test,f)

	with open('train_idx.pk','wb') as f:
		pickle.dump(train_idx,f)

	with open('test_idx.pk','wb') as f:
		pickle.dump(test_idx,f)

	print('user_num: ',len(data_train.keys()))
	#minMax = MinMaxScaler()
	#long_term_feature = minMax.fit_transform(np.array(long_term_feature))

	with open('long_term.pk','wb') as f:
		pickle.dump(long_term,f)

	#with open('long_term_feature.pk','wb') as f:
	#	pickle.dump(long_term_feature,f)

	def dataloader(X,X_cat,Y,hour,userid,hour_pre,week_pre):
		
		torch_dataset = Data.TensorDataset(torch.LongTensor(X),torch.LongTensor(X_cat),torch.LongTensor(Y),torch.LongTensor(hour),torch.LongTensor(userid),torch.LongTensor(hour_pre),torch.LongTensor(week_pre))
		loader = Data.DataLoader(
			dataset = torch_dataset,  
			batch_size = batch_size,  
			shuffle = True,
			num_workers = 0,
		)
		return loader

	loader_train = dataloader(train_x,train_x_cat,train_y,train_hour,train_userid,train_hour_pre,train_week_pre)
	loader_test = dataloader(test_x,test_x_cat,test_y,test_hour,test_userid,test_hour_pre,test_week_pre)

	pre_data = {}
	pre_data['size'] = [vocab_size_poi,vocab_size_cat,vocab_size_user,len(train_x),len(test_x)]
	pre_data['loader_train'] = loader_train
	pre_data['loader_test'] = loader_test

	with open('pre_data.txt','wb') as f:
		pickle.dump(pre_data,f)
	return pre_data
