#-*- coding:utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import pickle
from function import get_tensors,sentences_size,get_user_info,movies,users,movies_orig,users_orig,ratings

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}
# 用户ID转下标的字典
userid2idx = {val[0]:i for i, val in enumerate(users.values)}

#　用户信息
user_gender = {'M':'男性','F':'女性'}
user_age = {1:"Under 18",18:  "18-24",25:  "25-34",35:  "35-44",45:  "45-49", 50:  "50-55",56:  "56+"}
user_occupation = {0:  "other" , 1:  "academic/educator",2:  "artist",
                   3:  "clerical/admin",4:  "college/grad student",5:  "customer service",
                   6:  "doctor/health care",7:  "executive/managerial",8:  "farmer",
                   9:  "homemaker",10:  "K-12 student",11:  "lawyer",
                   12:  "programmer",13:  "retired",14:  "sales/marketing",
                   15:  "scientist",16:  "self-employed",17:  "technician/engineer",
                   18:  "tradesman/craftsman",19:  "unemployed",20:  "writer"}


# 打分情况
def rating_movie(user_id_val, movie_id_val):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载训练好的模型
        load_dir = pickle.load(open('./model/params.p', mode='rb'))
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, _, __ = get_tensors(loaded_graph)  # loaded_graph

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]
        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,
            movie_titles: titles,
            dropout_keep_prob: 1}

        # 获取预测值
        inference_val = sess.run([inference], feed)

        try:
            index = ratings[(ratings.UserID==user_id_val)&(ratings.MovieID==movie_id_val)].index.tolist()
            result = ratings['ratings'][index[0]]
        except:
            result = int(np.round(inference_val[0][0][0]))

        return result

# 相似推荐
def recommend_same_type_movie(movie_id_val, max_mov, top_k=20):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        recommend = []
        # 加载训练好的模型
        load_dir = pickle.load(open('./model/params.p', mode='rb'))
        movie_matrics = pickle.load(open('./model/movie_matrics.p', mode='rb'))

        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # 推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())

        recommend.append('-------------------------相似推荐-------------------------\n')
        recommend.append('您观看的电影是：\n')
        recommend.append('电影编号： %-6s电影类型： %s'% (str(movies_orig[movieid2idx[movie_id_val]][0]), movies_orig[movieid2idx[movie_id_val]][2]))
        recommend.append('电影名字： %s\n'% movies_orig[movieid2idx[movie_id_val]][1])
        recommend.append('---------------------------------------------------------')
        recommend.append('为您推荐%i部电影： \n' % max_mov)

        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)

        results = set()
        while len(results) != max_mov:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            recommend.append('电影编号： %-6s电影类型： %s' % (str(movies_orig[val][0]), movies_orig[val][2]))
            recommend.append('电影名字： %s\n' %  movies_orig[val][1])

        return recommend

# 根据用户信息推荐电影(猜你喜欢)
def recommend_user_favorite_movie(user_id_val, max_mov, top_k=10):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        recommend = []
        # 加载训练好的模型
        load_dir = pickle.load(open('./model/params.p', mode='rb'))
        movie_matrics = pickle.load(open('./model/movie_matrics.p', mode='rb'))
        users_matrics = pickle.load(open('./model/users_matrics.p', mode='rb'))

        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 推荐用户喜欢的电影
        probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        user_info = get_user_info(users_orig[userid2idx[user_id_val]])

        recommend.append('-------------------------猜你喜欢-------------------------\n')
        recommend.append('用户是：\n')
        recommend.append('用户编号： %-6s用户年龄： %s'% (user_info[0],user_info[2]))
        recommend.append('用户性别： %-4s用户职业： %s'% (user_info[1],user_info[3]))
        recommend.append('---------------------------------------------------------')
        recommend.append('为该用户推荐%i部电影： \n' % max_mov)

        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != max_mov:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            recommend.append('电影编号： %-6s电影类型： %s' % (str(movies_orig[val][0]), movies_orig[val][2]))
            recommend.append('电影名字： %s\n' %  movies_orig[val][1])

        return recommend

# 志同道合
def recommend_other_favorite_movie(movie_id_val, max_user, max_mov, top_k=20):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        recommend = []
        # 加载训练好的模型
        load_dir = pickle.load(open('./model/params.p', mode='rb'))
        movie_matrics = pickle.load(open('./model/movie_matrics.p', mode='rb'))
        users_matrics = pickle.load(open('./model/users_matrics.p', mode='rb'))

        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0]

        recommend.append('-------------------------志同道合-------------------------\n')
        recommend.append('您观看的电影是：\n')
        recommend.append('电影编号： %-6s电影类型： %s'% (str(movies_orig[movieid2idx[movie_id_val]][0]), movies_orig[movieid2idx[movie_id_val]][2]))
        recommend.append('电影名字： %s\n'% movies_orig[movieid2idx[movie_id_val]][1])
        recommend.append('---------------------------------------------------------')
        recommend.append('推荐好友，他们也在看： \n')
        count_user = 0
        for u in users_orig[favorite_user_id - 1]:
            user_info = get_user_info(u)
            recommend.append('用户编号： %-6s用户年龄： %s\n用户性别： %-4s用户职业： %s' % (user_info[0], user_info[2],user_info[1], user_info[3]))
            count_user += 1
            if count_user == max_user:
                break

        probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        p = np.argmax(sim, 1)
        recommend.append('---------------------------------------------------------')
        recommend.append('\n喜欢看这个电影的人还喜欢看： \n')

        results = set()
        for _ in range(len(p)):
            c = p[random.randrange(len(p))]
            results.add(c)
            if len(results) == max_mov:
                break
        for val in (results):
            recommend.append('电影编号： %-6s电影类型： %s' % (str(movies_orig[val][0]), movies_orig[val][2]))
            recommend.append('电影名字： %s\n' %  movies_orig[val][1])

        return recommend

