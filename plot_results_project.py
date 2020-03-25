#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import keras as keras
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import scipy as sp
from keras.models import Model
from keras import backend as K
import scipy


# ---------------------------- EVALUATION PLOTS ----------------------------

# normal loss plot
def plot_loss_results(History):

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();


# fancy plots
def plot_acc(History, n_epoch):

    # Check the history

    plt.figure(facecolor='white')

    # accuracy -------------------------------------------------------------------------
    ax1 = plt.subplot(2,1,1)

    plt.plot([x * 100 for x in History.history['accuracy']], label="acc", color="blue")
    plt.plot([x * 100 for x in History.history['val_accuracy']], label="val_acc", color="red")

    plt.title('Accuracy History')
    plt.ylabel('accuracy')
    # plt.xlabel('epoch')

    plt.legend(['train', 'valid'], loc='lower right')

    plt.ylim(0, 1)
    plt.xticks(np.arange(0, n_epoch + 1, 5))
    plt.yticks(np.arange(0, 100.1, 10))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    plt.grid()


def plot_loss(History, n_epoch):
    
    # loss -------------------------------------------------------------------------
    plt.subplot(2,1,2)

    plt.plot(History.history['loss'], label="loss", color="blue")
    plt.plot(History.history['val_loss'], label="val_loss", color="red")

    plt.title('Loss History')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'valid'], loc='lower left')

    plt.ylim(0)
    plt.xticks(np.arange(0, 50 + 1, 5))
    plt.grid()
    plt.show()

