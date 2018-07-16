from function import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    # 获取输入
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
    # 获取User的嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
    # 得到用户的特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    # 获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    # 获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    # 获取电影名的特征向量
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(dropout_keep_prob, movie_titles)
    # 得到电影的特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,movie_categories_embed_layer,dropout_layer)

    with tf.name_scope("inference"):
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)

    # 优化损失
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:
    grad_summaries = []
    for g, v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    loss_summary = tf.summary.scalar("loss", loss)
    print("Writing to {}\n".format(out_dir))

    # 训练
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Inference
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # 十折交叉验证
    kf = KFold(n_splits=10)
    count = 0
    target = []
    pred = []
    for epoch_i in range(num_epochs):
        for train_index, test_index in kf.split(features):
            count += 1
            train_X, test_X = features[train_index], features[test_index]
            train_y, test_y = targets_values[train_index], targets_values[test_index]
            train_batches = get_batches(train_X, train_y, batch_size)
            test_batches = get_batches(test_X, test_y, batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // batch_size):
                x, y = next(train_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,
                    movie_titles: titles,
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: dropout_keep,
                    lr: learning_rate}

                target,pred, step, train_loss, summaries, _ = sess.run([targets,inference,global_step, loss, train_summary_op, train_op], feed)

                losses['train'].append(train_loss)
                train_summary_writer.add_summary(summaries, step)

                # 每100次显示一次
                if (epoch_i * (len(train_X) // batch_size) + batch_i) % 100 == 0:
                    print('Training '+ str(count) +': Batch {:>4}/{}   train_loss = {:.3f}'.format(batch_i,(len(train_X) // batch_size), train_loss))

                # 每3500次显示一次
                if batch_i % 100 == 0 and batch_i != 0:
                    fpr, tpr, thresholds = roc_curve(target, pred, pos_label=2)
                    roc_auc = auc(fpr, tpr)

                    # ROC曲线
                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2,
                             label='ROC curve (AUC = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    plt.title('ROC curve')
                    plt.legend(loc="lower right")
                    plt.savefig('./result/Train_'+ str(count)+ '_' + str(epoch_i)+'.png')
                    # plt.show()

            # 使用测试数据进行迭代
            for batch_i in range(len(test_X) // batch_size):
                x, y = next(test_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,
                    movie_titles: titles,
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: 1,
                    lr: learning_rate}

                target, pred, step, test_loss, summaries = sess.run([targets,inference,global_step, loss, inference_summary_op], feed)

                # 保存测试损失
                losses['test'].append(test_loss)
                inference_summary_writer.add_summary(summaries, step)

                # 每100次显示一次
                if (epoch_i * (len(test_X) // batch_size) + batch_i) % 100 == 0:
                    print('Test ' + str(count) + ' : Batch {:>4}/{}   test_loss = {:.3f}'.format(batch_i, (len(test_X) // batch_size), test_loss))

                # 每300次显示一次
                if batch_i % 300 == 0 and batch_i != 0:
                    fpr, tpr, thresholds = roc_curve(target, pred, pos_label=2)
                    roc_auc = auc(fpr, tpr)

                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    plt.title('ROC curve')
                    plt.legend(loc="lower right")
                    plt.savefig('./result/Test_'+ str(count)+ '_' +str(epoch_i)+'.png')
                    # plt.show()

    # 保存模型
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

save_params((save_dir))
load_dir = load_params()

# 显示曲线图像
plt.figure(0)
plt.plot(losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()
# plt.show()
plt.savefig("./model/train_loss.jpg")

plt.figure(1)
plt.plot(losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()
# plt.show()
plt.savefig("./model/test_loss.jpg")

loaded_graph = tf.Graph()  #
movie_matrics = []
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存好的模型
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # 从模型中获取Tensors
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(loaded_graph)  #loaded_graph

    for item in movies.values:
        categories = np.zeros([1, 18])
        categories[0] = item.take(2)

        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)

        feed = {
            movie_id: np.reshape(item.take(0), [1, 1]),
            movie_categories: categories,
            movie_titles: titles,
            dropout_keep_prob: 1}

        movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
        movie_matrics.append(movie_combine_layer_flat_val)

pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('./model/movie_matrics.p', 'wb'))
movie_matrics = pickle.load(open('./model/movie_matrics.p', mode='rb'))

loaded_graph = tf.Graph()
users_matrics = []
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存好的模型
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # 从模型中获取Tensors
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __,user_combine_layer_flat = get_tensors(loaded_graph)  #loaded_graph

    for item in users.values:

        feed = {
            uid: np.reshape(item.take(0), [1, 1]),
            user_gender: np.reshape(item.take(1), [1, 1]),
            user_age: np.reshape(item.take(2), [1, 1]),
            user_job: np.reshape(item.take(3), [1, 1]),
            dropout_keep_prob: 1}

        user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
        users_matrics.append(user_combine_layer_flat_val)

pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('./model/users_matrics.p', 'wb'))
users_matrics = pickle.load(open('./model/users_matrics.p', mode='rb'))

print("End")