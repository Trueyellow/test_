# Test for optimal flow fusion
import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize

# keras module
from keras import optimizers, callbacks
from keras.models import *
from keras.layers.core import *
from keras.layers import merge, BatchNormalization, Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPool2D, Input, UpSampling2D
from keras.layers.merge import add, multiply, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical

# Local module
from attention_utils import get_activations, get_data
from utils import io

np.random.seed(1337)  # for reproducibility

input_dim = 8192
# input image dimensions
img_rows, img_cols = 256, 256
# number of lables
n_label = 16
case_per_sample = 16
input_shape = (1, img_rows, img_cols, 3)
model_path = 'stack_soft_attention_cnn_with_flow_osd'

# set before using
data_path = None
root = None

#
OSD_interface = io.OlympicIO(data_path, root)

def get_random_train_data_with_optimal():
    # get osd training data
    video = np.empty((0, 256, 256, 3))
    flow = np.empty((0, 256, 256, 3))
    label = np.empty((0, n_label))
    occurance = np.zeros((1, 16))

    for loop in range(50):
        print("Train Loop Number: ",loop)
        if len(OSD_interface.reduce_dict) == 0:
            video_tmp, label_tmp = OSD_interface.random_video(reset=True)
        else:
            video_tmp, label_tmp = OSD_interface.random_video()
        num_label = label_tmp
        if occurance[0, num_label] == 0:
            frame_tmp, flow_tmp, label_tmp, prev_tmp = OSD_interface.extract_features(video_tmp, label_tmp)
            video = np.concatenate((video, frame_tmp), axis=0)
            flow = np.concatenate((flow, flow_tmp), axis=0)
            label = np.concatenate((label, label_tmp), axis=0)
            occurance[0, num_label] += 1
        else:
            pass

    video = (video - 127.5) / 127.5
    return video, flow, label


def get_test_data_with_flow():
    video = np.empty((0, 256, 256, 3))
    flow = np.empty((0, 256, 256, 3))
    label = np.empty((0, n_label))
    test_list = list(OSD_interface.test_osd().items())
    for i in range(10):
        rand = random.randint(0, len(test_list)-1)
        video_name = test_list[rand][0]
        video_temp = cv2.VideoCapture(video_name)
        label_temp = test_list[rand][1]
        frame_tmp, flow_tmp, label_tmp, prev_tmp = OSD_interface.extract_features(video_temp, label_temp)
        video = np.concatenate((video, frame_tmp), axis=0)
        flow = np.concatenate((flow, frame_tmp), axis=0)
        label = np.concatenate((label, label_tmp), axis=0)
        print("Test loop number: ", i)

    video = (video - 127.5) / 127.5
    return video, flow, label


def feature_stack_model_base():
    '''
    :param input_shape: shape = (batch_size, time_steps, input_dim)
    :param frame_num: TimeDistributed batch size
    :return: model
    '''
    inputs = Input(shape=input_shape)
    print(inputs.shape)
    '''
    # forward 1
    '''
    conv1 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(inputs)
    batch1 = TimeDistributed(BatchNormalization())(conv1)
    conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(batch1)
    batch2 = TimeDistributed(BatchNormalization())(conv2)
    pool2 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch2)

    '''
    # conv_deconv 1
    '''
    cdconv1 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(inputs)
    batch3 = TimeDistributed(BatchNormalization())(cdconv1)
    pool3 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch3)

    cdconv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(pool3)
    batch4 = TimeDistributed(BatchNormalization())(cdconv2)
    pool4 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch4)

    cdconv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(pool4)
    batch5 = TimeDistributed(BatchNormalization())(cdconv3)
    pool5 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch5)

    cdconv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(pool5)
    batch6 = TimeDistributed(BatchNormalization())(cdconv4)
    pool6 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch6)

    cdconv5 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(pool6)
    batch7 = TimeDistributed(BatchNormalization())(cdconv5)
    pool7 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch7)

    cdconv6 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(pool7)
    batch8 = TimeDistributed(BatchNormalization())(cdconv6)
    us1 = TimeDistributed(UpSampling2D((2, 2)))(batch8)

    cdconv7 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(us1)
    batch9 = TimeDistributed(BatchNormalization())(cdconv7)
    us2 = TimeDistributed(UpSampling2D((2, 2)))(batch9)

    cdeconv8 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(us2)
    batch10 = TimeDistributed(BatchNormalization())(cdeconv8)
    us3 = TimeDistributed(UpSampling2D((2, 2)))(batch10)

    cdeconv9 = TimeDistributed(Conv2D(64, (3, 3), activation='sigmoid', padding='same'))(us3)
    batch11 = TimeDistributed(BatchNormalization())(cdeconv9)
    us4 = TimeDistributed(UpSampling2D((2, 2)))(batch11)  # attention mask

    cdeconv10 = TimeDistributed(Conv2D(64, (1, 1), activation='sigmoid', padding='same'))(us4)
    batch12 = TimeDistributed(BatchNormalization())(cdeconv10)

    '''
    # Attention Merge
    '''
    # soft attention block 1
    soft_att = multiply([pool2, batch12])
    after_soft_att = add([pool2, soft_att])
    batch13 = BatchNormalization()(after_soft_att)

    '''
    # forward 2
    '''

    conv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(batch13)
    batch13 = TimeDistributed(BatchNormalization())(conv3)
    pool8 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch13)

    conv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(pool8)
    batch14 = TimeDistributed(BatchNormalization())(conv4)
    pool9 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch14)

    conv5 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(pool9)
    batch15 = TimeDistributed(BatchNormalization())(conv5)
    pool10 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch15)

    '''
    # conv_deconv 2
    '''
    cdeconv11 = TimeDistributed(Conv2D(32, kernel_size=(3, 3),
                                       activation='relu', padding='same'))(batch13)
    batch15 = TimeDistributed(BatchNormalization())(cdeconv11)
    pool11 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch15)

    cdconv12 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(pool11)
    batch16 = TimeDistributed(BatchNormalization())(cdconv12)
    pool12 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch16)

    cdconv13 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(pool12)
    batch17 = TimeDistributed(BatchNormalization())(cdconv13)
    pool13 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch17)

    cdconv14 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(pool13)
    batch18 = TimeDistributed(BatchNormalization())(cdconv14)
    pool14 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch18)

    cdconv15 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(pool14)
    batch19 = TimeDistributed(BatchNormalization())(cdconv15)
    pool15 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch19)

    cdconv16 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(pool15)
    batch20 = TimeDistributed(BatchNormalization())(cdconv16)
    us5 = TimeDistributed(UpSampling2D((2, 2)))(batch20)

    cdconv17 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(us5)
    batch21 = TimeDistributed(BatchNormalization())(cdconv17)
    us6 = TimeDistributed(UpSampling2D((2, 2)))(batch21)

    deconv18 = TimeDistributed(Conv2D(256, (1, 1), activation='sigmoid', padding='same'))(us6)
    batch22 = TimeDistributed(BatchNormalization())(deconv18)

    '''
    # Attention Merge
    '''
    # soft attention block 1
    soft_att_2 = multiply([pool10, batch22])
    after_soft_att_2 = add([pool10, soft_att_2])
    batch23 = BatchNormalization()(after_soft_att_2)

    '''
    # forward 3
    '''
    conv6 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(batch23)
    batch23 = TimeDistributed(BatchNormalization())(conv6)
    pool16 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch23)

    conv7 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(pool16)
    batch24 = TimeDistributed(BatchNormalization())(conv7)
    pool17 = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(batch24)

    reshape_flatten = TimeDistributed(Flatten())(pool17)
    branch = LSTM(1024, return_sequences=True)(reshape_flatten)
    return inputs, branch


def two_model_fusion_to_one_with_attention(branch_1, input_1, branch_2, input_2, frame_num):

    # in order to build the model without time_distributed_number( is None when we are building model),
    # I simulate stack the layers as if there are time_distributed_number shape

    merged_1_simu_time = branch_1
    merged_2_simu_time = branch_2

    print(frame_num)
    print(branch_1.shape, branch_2.shape)
    # check the shape to see if we need to stack the layers
    if branch_1.shape[1] is not frame_num:
        print(branch_1.shape)
        for _ in range(frame_num - 1):
            merged_1_simu_time = concatenate([merged_1_simu_time, branch_1], axis=1)
            merged_2_simu_time = concatenate([merged_2_simu_time, branch_2], axis=1)

    merged = add([merged_1_simu_time, merged_2_simu_time])
    #----------------------------------------- lstm attention ---------------------------------------------------------

    # steps is assigned in the model building, and is based
    # on TimeDistributed batch size
    steps = frame_num

    single = True

    input_dim = int(merged.shape[2])
    print(input_dim)

    a_lstm_1 = LSTM(input_dim, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(merged)
    a_l2_norm = Lambda(lambda x : K.l2_normalize(x, axis=2), name='normalization')(a_lstm_1)
    a_l2_permute_1 = Permute((2, 1))(a_l2_norm)

    """
    I have checked the original github project and this reshape part is used to check
    whether the Permuted layer has the same shape as we want, since we can only get
    the shape when training, I comment this line so that our model can successfully build.
    """
    #a_reshape = Reshape((input_dim, steps))(a_l2_permute_1)

    a_dense_1 = Dense(steps)(a_l2_permute_1)
    a_batch_norm = BatchNormalization()(a_dense_1)
    a_dropout = Dropout(0.2)(a_batch_norm)
    repeat_vector = Dense(steps, activation='sigmoid')(a_dropout)

    if single:
        a_dim_reduction = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(repeat_vector)
        repeat_vector = RepeatVector(input_dim)(a_dim_reduction)

    attention_prob = Permute((2, 1), name='attention_vec')(repeat_vector)
    output_attention_mul = add([merged, multiply([merged, attention_prob], name='attention_mul')], name='soft_assign')

#----------------------------------------- lstm attention ---------------------------------------------------------

    """
    This code should be checked again since I changed the "attention_out_dense_1"'s
    layer shape to fit the next :output" layer's shape
    """
    attention_out_batch_norm = BatchNormalization()(output_attention_mul)
    attention_out_dense_1 = Dense(256)(attention_out_batch_norm)
    attention_out_dropout = Dropout(0.3)(attention_out_dense_1)
    output = Dense(n_label, activation='softmax')(attention_out_dropout)

    model = Model(inputs=[input_1, input_2], outputs=output)
    return model


def main():
    TIME_DISTRIBUTE_BATCH = 16

    input_1, branch_1 = feature_stack_model_base()
    input_2, branch_2 = feature_stack_model_base()
    fusion_model = two_model_fusion_to_one_with_attention(branch_1, \
                                        input_1, branch_2, input_2, TIME_DISTRIBUTE_BATCH)

    opt = optimizers.adam(lr=0.0001)
    fusion_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    N = 10
    for loop in range(N):
        print("Loop: ", loop)
        # frame, label, first_half_done = get_training_data_osd(first_half_done)
        frame, flow, label = get_random_train_data_with_optimal()
        frame = np.expand_dims(frame, axis=1)
        flow = np.expand_dims(flow, axis=1)
        test_frame, test_flow, test_label = get_test_data_osd()
        test_frame = np.expand_dims(test_frame, axis=1)
        test_flow = np.expand_dims(test_flow, axis=1)
        print("unique label", np.unique([np.where(tmp == 1)[0][0] for tmp in label]))
        m.fit([frame, flow], label, epochs=2, batch_size=TIME_DISTRIBUTE_BATCH)
        m.reset_states()
        score = m.evaluate([test_frame, test_flow], test_label, batch_size=TIME_DISTRIBUTE_BATCH)
        m.reset_states()
        print(score)
        m.save(model_path)


if __name__ == '__main__':
    main()
