from os import listdir, path, makedirs
import random
import sys
import time
import datetime
import pandas as pd

def print_stats(data):
  total_ratings = 0
  print("STATS")
  for user in data:
    total_ratings += len(data[user])
  print("Total Ratings: {}".format(total_ratings))
  print("Total User count: {}".format(len(data.keys())))

def save_data_to_file(data, filename):
  with open(filename, 'w') as out:
    for userId in data:
      for record in data[userId]:
        out.write("{}\t{}\t{}\n".format(userId, record[0], record[1]))

def create_data_timesplit(all_data,
                                  train_min,
                                  train_max,
                                  test_min,
                                  test_max):
  """
  Creates time-based split of data into train, and (validation, test)
  :param all_data:
  :param train_min:
  :param train_max:
  :param test_min:
  :param test_max:
  :return:
  """
  train_min_ts = time.mktime(datetime.datetime.strptime(train_min,"%Y-%m-%d").timetuple())
  train_max_ts = time.mktime(datetime.datetime.strptime(train_max, "%Y-%m-%d").timetuple())
  test_min_ts = time.mktime(datetime.datetime.strptime(test_min, "%Y-%m-%d").timetuple())
  test_max_ts = time.mktime(datetime.datetime.strptime(test_max, "%Y-%m-%d").timetuple())

  training_data = dict()
  validation_data = dict()
  test_data = dict()

  train_set_items = set()

  for userId, userRatings in all_data.items():
    time_sorted_ratings = sorted(userRatings, key=lambda x: x[2])  # sort by timestamp
    for rating_item in time_sorted_ratings:
      if rating_item[2] >= train_min_ts and rating_item[2] <= train_max_ts:
        if not userId in training_data:
          training_data[userId] = []
        training_data[userId].append(rating_item)
        train_set_items.add(rating_item[0]) # keep track of items from training set
      elif rating_item[2] >= test_min_ts and rating_item[2] <= test_max_ts:
        if not userId in training_data: # only include users seen in the training set
          continue
        if not userId in test_data:
            test_data[userId] = []
        test_data[userId].append(rating_item)
        p = random.random()
        if p <=0.3:
          if not userId in validation_data:
            validation_data[userId] = []
          validation_data[userId].append(rating_item)
        else:
          pass



  for userId, userRatings in validation_data.items():
    validation_data[userId] = [rating for rating in userRatings if rating[0] in train_set_items]

  return training_data, validation_data, test_data



def load_data(file_name = 'train.csv'):
    path = '../../data/{}'.format(file_name)
    df = pd.read_csv(path)
    return df

def process_df(df,type_ ='train'):
    list_ids =list( df.Id)
    preds = list(df['Prediction'])
    ts = None
    if(type_ == 'train'):
        ts = int(time.mktime(datetime.datetime.strptime("2004-01-01","%Y-%m-%d").timetuple()))
    else:
        ts = int(time.mktime(datetime.datetime.strptime("2005-12-20","%Y-%m-%d").timetuple()))
    splitted = [a.split('_') for a in list_ids]
    users = [split[0][1:] for split in splitted]
    movies = [split[1][1:] for split in splitted]
    all_data = {}
    for i in range(len(preds)):
        if users[i] not in all_data.keys():
            all_data[users[i]] = []
        
        all_data[users[i]].append([movies[i],preds[i],ts]);
    
    return all_data


def combine_train_test():
    all_train = process_df(load_data());
    all_test = process_df(load_data('test.csv'),type_='test')
    
    for user, test_r in all_test.items():
        all_train[user].extend(test_r)    
    
    return all_train


def main(args):
 
  out_folder = './data_processed'
  all_data = combine_train_test()
  print("STATS FOR ALL INPUT DATA")
  print_stats(all_data)

  (nf_train, nf_valid, nf_test) = create_data_timesplit(all_data,
                                                                "1999-12-01",
                                                                "2005-11-30",
                                                                "2005-12-01",
                                                                "2005-12-31")
  print(" full train")
  print_stats(nf_train)
  save_data_to_file(nf_train, out_folder + "/TRAIN/nf.train.txt")
  print(" full valid")
  print_stats(nf_valid)
  save_data_to_file(nf_valid, out_folder + "/VALID/nf.valid.txt")
  print(" full test")
  print_stats(nf_test)
  save_data_to_file(nf_test, out_folder + "/TEST/nf.test.txt")



if __name__ == "__main__":
    main(sys.argv)

