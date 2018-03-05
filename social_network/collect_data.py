import json
import sys
import re
from os import listdir
from bz2 import BZ2File as bzopen
from time import time
from collections import Counter
import pandas as pd
import nltk
from nltk import word_tokenize

# locate comment files and thread files
comment_dir = '../jiaqima/BlockChain/'
comment_files = [f for f in listdir(comment_dir) if '.bz2' in f]
unique_months = [f[:f.index('.')] for f in comment_files]

print(unique_months)



print('\nVariable initializing...')
comments = {'comment_id':[], 'length': [], 'month': [], 'subreddit': []}
temp_threads = {} # {thread_id: {'body': [comment1, comment2], 'month':[month1, month2], 'users': [user1, user2...] }}
threads = {'thread_id': [], 'length':[], 'num_user':[], 'num_comment':[]}

temp_users = {} # {user_id: {month: {'thread_id':[], 'num_comment':0, 'subreddit': []}}}
users = {'user_id': []}

temp_subreddit = {} # {subreddit: {month: {'user':[], 'thread':[], 'num_comment': 0}}}
subreddits = {'subreddit': []}

for m in unique_months:
	threads[m] = []

	users[m+'_num_thread'] = []
	users[m+'_num_comment'] = []
	users[m+'_num_subreddit'] = []

	subreddits[m+'_num_thread'] = []
	subreddits[m+'_num_comment'] = []
	subreddits[m+'_num_user'] = []
	subreddits[m+'_num_unique_user'] = []




for f in comment_files:
	with bzopen(comment_dir + f, 'rb') as bzfin:
		month = f[:f.index('.')]
		print("\nStart working on month:", month)

		i = 0
		for line in bzfin.readlines():
			line = line.decode('utf-8')
			line = json.loads(line)

			if i % 10000 == 0:
				print(i, 'Start extracting attributes of COMMENTS (comments))')
			# comments attributes
			comment_id = line['id']
			comment = line['body']
			tokens = word_tokenize(comment)
			subreddit = line['subreddit']

			comments['comment_id'].append(comment_id)
			comments['length'].append(len(tokens))
			comments['month'].append(month)
			comments['subreddit'].append(subreddit)

			if i % 10000 == 0:
				print(i, 'Start extracting attributes of THREADS (temp_threads)')
			# threads attributes
			thread_id = line['link_id']
			if thread_id not in temp_threads:
				temp_threads[thread_id] = {'body': [], 'month':[], 'user': []}
			user = line['author']
			temp_threads[thread_id]['body'].append(comment)
			temp_threads[thread_id]['month'].append(month)
			temp_threads[thread_id]['user'].append(user)

			if i % 10000 == 0:
				print(i, 'Start extracting attributes of USER (temp_users)')
			if user not in temp_users:
				temp_users[user] = {}
			if month not in temp_users[user]:
				temp_users[user][month] = {'thread_id':[], 'num_comment':0, 'subreddit': []}

			user_month_dict = temp_users[user][month]
			if thread_id not in user_month_dict['thread_id']:
				user_month_dict['thread_id'].append(thread_id)
			if subreddit not in user_month_dict['subreddit']:
				user_month_dict['subreddit'].append(subreddit)
			user_month_dict['num_comment'] += 1

			if i % 10000 == 0:
				print(i, 'Start extracting attributes of SUBREDDIT (temp_subreddit)')
			if subreddit not in temp_subreddit:
				temp_subreddit[subreddit] = {}
			if month not in temp_subreddit[subreddit]:
				temp_subreddit[subreddit][month] = {'user':[], 'thread':[], 'num_comment': 0}

			subreddit_month = temp_subreddit[subreddit][month]
			subreddit_month['user'].append(user)
			if thread_id not in subreddit_month['thread']:
				subreddit_month['thread'].append(thread_id)
			subreddit_month['num_comment'] += 1

			i += 1


print('\n1. Assembling COMMENTS DataFrame (comments_df)...')
comments_df = pd.DataFrame(comments)
comments_df.to_csv('comment_eda_data.tsv', index=False, sep='\t')


print('2. Assembling THREADS DataFrame (threads_df)...')
for thread_id, content in temp_threads.items():

	# comments related attributes
	body = content['body']
	length = 0
	for comment in body:
		tokens = word_tokenize(comment)
		length += len(tokens)

	# month 
	month = content['month']
	month_count = Counter(month)


	threads['thread_id'].append(thread_id)
	threads['length'].append(length)
	threads['num_user'].append(len(content['user']))
	threads['num_comment'].append(len(body))
	for m in unique_months:
		count = month_count.get(m, 0)
		threads[m].append(count)
threads_df = pd.DataFrame(threads)
threads_df.to_csv('thread_eda_data.tsv', index=False, sep='\t')



print('3. Assembling USERS DataFrame (user_df)...')
for user_id, month_content in temp_users.items():
	users['user_id'].append(user_id)
	for month, content in month_content.items():
		num_thread = len(content['thread_id'])
		num_subreddit = len(content['subreddit'])
		users[month+'_num_thread'].append(num_thread)
		users[month+'_num_comment'].append(content['num_comment'])
		users[month+'_num_subreddit'].append(num_subreddit)
	
user_df = pd.DataFrame(users)
user_df.to_csv('user_eda_data.tsv', index=False, sep='\t')

print('4. Assembling SUBREDDIT DataFrame (subreddit_df)...')
for subreddit_name, month_content in temp_subreddit.items():
	subreddits['subreddit'].append(subreddit_name)
	for month, content in month_content.items():
		num_thread = len(content['thread'])
		num_comment = content['num_comment']
		num_user = len(content['user'])
		num_unique_user = len(set(content['user']))


		subreddits[month+'_num_thread'].append(num_thread)
		subreddits[month+'_num_comment'].append(num_comment)
		subreddits[month+'_num_user'].append(num_user)
		subreddits[month+'_num_unique_user'].append(num_unique_user)
subreddit_df = pd.DataFrame(subreddits)
subreddit_df.to_csv('subreddit_eda_data.tsv', index=False, sep='\t')

# print(comments_df)



