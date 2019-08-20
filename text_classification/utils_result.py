import os
import keras
import shutil
import pydot as pyd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils import get_proper_callback
from sklearn.utils.extmath import cartesian
from keras.utils import vis_utils
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model



def display_model_history(history, model_name, ax):
    keys_nd = np.array(list(history.keys())).reshape(ax.shape[0], ax.shape[1])
    for i, j in cartesian((np.arange(ax.shape[0]), np.arange(ax.shape[1]))):
        key = keys_nd[i][j]
        ax[i][j].plot(history[key], label=model_name)
        ax[i][j].title.set_text(key)
        ax[i][j].set_xlabel("epoch")
        ax[i][j].set_ylabel("metric_value")
        ax[i][j].grid(True)
        ax[i][j].legend()

        
def display_results(markered_path, save_img=True, comp_name="charlevel"):
    history_dict = {}
    names = []
    for filename in os.listdir(markered_path):
        if "_history.csv" in filename:
            df = pd.read_csv(os.path.join(markered_path, filename)).drop(labels="epoch", axis=1)
            history_dict[filename[:-len("_history.csv")]] = df
            ncols = (df.columns.shape[0]) // 2
    
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(20,10))
    for model_name in history_dict:
        names.append(model_name)
        display_model_history(history=history_dict[model_name], model_name=model_name, ax=ax)
        
           
    if save_img:
        if len(names) == 1:
            fig.savefig("{}/{}_history_figure.png".format(markered_path, names[0]), dpi=fig.dpi)
        else:
            fig.savefig("{}/{}_history_figure.png".format(markered_path, comp_name), dpi=fig.dpi)
        

def save_report(model, model_name, markered_path):
    vis_utils.pydot = pyd        
    display_results(markered_path, save_img=True)
    vis_utils.plot_model(model, "{}/{}_vizu.png".format(markered_path, model_name), show_layer_names=False, show_shapes=True)
    with open('{}/{}_report.txt'.format(markered_path, model_name), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    model.load_weights(os.path.join(markered_path, "saves/weights.h5"))
    return model

        
def update_fit_params(fit_params, model_name, tensorboard_params=None):
    markered_path = "about_models/{}/{}".format(model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(markered_path)
    os.makedirs(os.path.join(markered_path, "saves"))
    fit_params["callbacks"] = [callbacks.EarlyStopping(patience=5), callbacks.CSVLogger(os.path.join(markered_path, "{}_history.csv".format(model_name))),
                              callbacks.ModelCheckpoint(filepath=os.path.join(markered_path, "saves/weights.h5"), save_best_only=True, save_weights_only=True)]
    if tensorboard_params:
        tensorboard_params["log_dir"] = os.path.join(markered_path, "logs")
        fit_params["callbacks"] += [get_proper_callback(callbacks.TensorBoard, tensorboard_params)]
    return fit_params, markered_path


def evaluation(model, x_test, y_test, model_name, test_result_dict):
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    test_result_dict[model_name] = (y_test_pred == y_test).mean()
    print("Wynik na zbiorze testowym modelu {}, to {:.4f}".format(model_name, test_result_dict[model_name]))
    return test_result_dict


def update_common_history_folder(base_folder):
    base_folder = "about_models"
    assert(os.path.exists(base_folder))
    common_folder = os.path.join(base_folder, "comparision")
    if not os.path.exists(common_folder):
        os.makedirs(common_folder)

    for modelname in os.listdir(base_folder):
        if "model" not in modelname:
            continue
        last_checkpoint = max(os.listdir(os.path.join(base_folder, modelname)))
        for filename in os.listdir(os.path.join(base_folder, modelname, last_checkpoint)):
            if "_history.csv" in filename:
                filepath = os.path.join(base_folder, modelname, last_checkpoint, filename)
                shutil.copy2(filepath, os.path.join(common_folder, filename))
