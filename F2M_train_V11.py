# -*- coding:utf-8 -*-
from F2M_model_V10 import *
from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,

                           "tar_size": 256,

                           "tar_load_size": 276,
                           
                           "batch_size": 2,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",
                           
                           "B_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_40_63_16_39/train/male_16_39_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_16_39/",

                           "age_range": [40, 64],

                           "n_classes": 256,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Pictures/img",
                           
                           "A_test_txt_path": "",
                           
                           "A_test_img_path": "",
                           
                           "B_test_txt_path": "",
                           
                           "B_test_img_path": "",
                           
                           "test_dir": "A2B",
                           
                           "fake_B_path": "",
                           
                           "fake_A_path": ""})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_load_size, FLAGS.tar_load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    B_img = B_img / 127.5 - 1.

    B_lab = int(B_data[1])
    A_lab = int(A_data[1])

    return A_img, A_lab, B_img, B_lab

def te_input_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    lab = lab

    return img, lab

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def decreas_func(x):
    return tf.maximum(0, tf.math.exp(x * (-2.77 / 100)))

def increase_func(x):
    x = tf.cast(tf.maximum(1, x), tf.float32)
    return tf.math.log(x + 1e-7)

def cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
             A_batch_images, B_batch_images, B_batch_labels, A_batch_labels):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_B, real_A_mid = model_out(A2B_G_model, A_batch_images, True)
        fake_A_, fake_B_mid = model_out(B2A_G_model, fake_B, True)

        fake_A, real_B_mid = model_out(B2A_G_model, B_batch_images, True)
        fake_B_, fake_A_mid = model_out(A2B_G_model, fake_A, True)

        # identification    # 이것도 추가하면 괜찮지 않을까?
        #id_fake_A, _ = model_out(B2A_G_model, A_batch_images, True)
        #id_fake_B, _ = model_out(A2B_G_model, B_batch_images, True)

        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, fake_B, True)
        DA_real = model_out(A_discriminator, A_batch_images, True)
        DA_fake = model_out(A_discriminator, fake_A, True)

        ################################################################################################
        # 나이에 대한 distance를 구하는곳
        return_loss = 0.
        for i in range(FLAGS.batch_size):   # 기존의 데이터로하려면 compare label을 하나 더 만들어야 한다 기억해!!!!
            energy_ft = tf.reduce_sum(tf.abs(fake_B_mid[i] - real_B_mid), 1)
            energy_ft2 = tf.reduce_sum(tf.abs(fake_A_mid[i] - real_A_mid), 1)   # 이걸 새로운 compare label에 대입해야함
            
            compare_label = tf.subtract(B_batch_labels, B_batch_labels[i]) # real_B_mid fake_B_mid
            compare_label2 = tf.subtract(A_batch_labels, A_batch_labels[i])  # real_A_mid fake_A_mid

            T = 4
            label_buff = tf.less(tf.abs(compare_label), T)
            label_cast = tf.cast(label_buff, tf.float32)

            label_buff2 = tf.less(tf.abs(compare_label2), T)
            label_cast2 = tf.cast(label_buff2, tf.float32)

            realB_fakeB_loss = label_cast * increase_func(energy_ft) \
                + (1 - label_cast) * 5 * decreas_func(energy_ft)

            realA_fakeA_loss = label_cast2 * increase_func(energy_ft2) \
                + (1 - label_cast2) * 5 * decreas_func(energy_ft2)

            # A와 B 나이가 다르면 감소함수, 같으면 증가함수

            loss_buf = 0.
            for j in range(FLAGS.batch_size):
                loss_buf += realB_fakeB_loss[j] + realA_fakeA_loss[j]
            loss_buf /= FLAGS.batch_size

            return_loss += loss_buf
        return_loss /= FLAGS.batch_size
        ################################################################################################

        #id_loss = tf.reduce_mean(tf.abs(id_fake_A - A_batch_images)) * 10.0 + tf.reduce_mean(tf.abs(id_fake_B - B_batch_images)) * 10.0

        Cycle_loss = (tf.reduce_mean(tf.abs(fake_A_ - A_batch_images))) * 10.0 + (tf.reduce_mean(tf.abs(fake_B_ - B_batch_images))) * 10.0
        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) + tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2. \
            + (tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + return_loss
        d_loss = Adver_loss

    g_grads = g_tape.gradient(g_loss, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables)
    d_grads = d_tape.gradient(d_loss, A_discriminator.trainable_variables + B_discriminator.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_discriminator.trainable_variables + B_discriminator.trainable_variables))

    return g_loss, d_loss

def main():

    A2B_G_model = F2M_generator_V2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_G_model = F2M_generator_V2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_discriminator = F2M_discriminator(input_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))
    A_discriminator = F2M_discriminator(input_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))

    A2B_G_model.summary()
    B_discriminator.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   B_discriminator=B_discriminator,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):
            min_ = min(len(A_images), len(B_images))
            A = list(zip(A_images, A_labels))
            B = list(zip(B_images, B_labels))
            shuffle(B)
            shuffle(A)
            B_images, B_labels = zip(*B)
            A_images, A_labels = zip(*A)
            A_images = A_images[:min_]
            A_labels = A_labels[:min_]
            B_images = B_images[:min_]
            B_labels = B_labels[:min_]

            A_zip = np.array(list(zip(A_images, A_labels)))
            B_zip = np.array(list(zip(B_images, B_labels)))

            # 가까운 나이에 대해서 distance를 구하는 loss를 구성하면, 결국에는 해당이미지의 나이를 그대로 생성하는 효과?를 볼수있을것
            gener = tf.data.Dataset.from_tensor_slices((A_zip, B_zip))
            gener = gener.shuffle(len(B_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = min_ // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, A_batch_labels, B_batch_images, B_batch_labels = next(train_it)

                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
                                          A_batch_images, B_batch_images, B_batch_labels, A_batch_labels)

                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))
                
                if count % 100 == 0:
                    fake_B, _ = model_out(A2B_G_model, A_batch_images, False)
                    fake_A, _ = model_out(B2A_G_model, B_batch_images, False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), fake_B[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/fake_A_{}.jpg".format(count), fake_A[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0] * 0.5 + 0.5)


                #if count % 1000 == 0:
                #    num_ = int(count // 1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} folder to store the weight!".format(num_))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                #                               A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                #                               g_optim=g_optim, d_optim=d_optim)
                #    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)

                count += 1

    else:
        if FLAGS.test_dir == "A2B":
            A_train_data = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
            A_train_data = [FLAGS.A_img_path + data for data in A_train_data]
            A_train_label = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            A_test_data = np.loadtxt(FLAGS.A_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            A_test_data = [FLAGS.A_test_img_path + data for data in A_test_data]
            A_test_label = np.loadtxt(FLAGS.A_test_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((A_train_data, A_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((A_test_data, A_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(A_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(A_test_data) // 1

            for i in range(te_idx):
                te_A_images, te_A_labels = next(te_it)
                fake_B, te_feature = model_out(A2B_G_model, te_A_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_A_images, tr_A_labels = next(tr_it)
                    _, tr_feature = model_out(A2B_G_model, tr_A_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_A_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (A_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_B_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_B[0].numpy() * 0.5 + 0.5)



        if FLAGS.test_dir == "B2A":
            B_train_data = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
            B_train_data = [FLAGS.B_img_path + data for data in B_train_data]
            B_train_label = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            B_test_data = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            B_test_data = [FLAGS.B_test_img_path + data for data in B_test_data]
            B_test_label = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((B_train_data, B_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((B_test_data, B_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(B_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(B_test_data) // 1

            for i in range(te_idx):
                te_B_images, te_B_labels = next(te_it)
                fake_A, te_feature = model_out(B2A_G_model, te_B_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_B_images, tr_B_labels = next(tr_it)
                    _, tr_feature = model_out(B2A_G_model, tr_B_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_B_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (B_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_A_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_A[0].numpy() * 0.5 + 0.5)



if __name__ == "__main__":
    main()
