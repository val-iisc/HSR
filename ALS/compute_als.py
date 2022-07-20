import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt 
import pickle as pkl
from tqdm import tqdm
import json


n_attr = 5
seq_len = 11


# this module will compute the delta between the interpolated attribute differences and the real predictions of attributes scores from the attr classfs
def create_attr_ppl(attrs, logits_stds, logits_means):
    # print("attr shape: ", attrs.shape)

    image_t0 = 0       # Taking the first image id
    image_t1 = attrs.shape[0]-1      # Taking the last image id for interpolation 

    n_total = attrs.shape[0]
    
    start_attr = attrs[image_t0, ...].copy()
    end_attr = attrs[image_t1, ...].copy() 

    deltas = []
    deltas_true = [] # For testing 
    for t in range(0, n_total): 
        t_normed = t / (n_total -1)

        # Interpolating the attributes from the first and the end attribute 
        attr_interpolate = start_attr * (1-t_normed) + end_attr * (t_normed)  

        # Extracting the attributes scores for an intermediate image to be checked
        w_interpolate = attrs[t, ...].copy()


        # Subtracting the means
        attr_interpolate = attr_interpolate - logits_means
        w_interpolate = w_interpolate - logits_means 

        delta = np.divide((attr_interpolate - w_interpolate), logits_stds) 
        
        # delta = np.divide(w_interpolate, logits_stds)
        deltas.append(delta)  


    return deltas


# Computing the metrics required for attribute score differences for a set of images 
def compute_attr_ppl_metrics(img_id_list, attrs, logits_vars):
    deltas = create_attr_ppl(img_id_list, attrs, logits_vars) 

    deltas_array = np.array(deltas)
    deltas_ppl = np.mean(deltas_array, axis=1)

    # Taking the average of all the deltas in the given image sequence 
    delta_acc = np.sum(deltas_ppl) 
    delta_max = np.max(deltas_ppl)

    print("delta acc: ", delta_acc)
    print("delta max: ", delta_max)

    return delta_acc, delta_max


def compute_stats(fld_path):
    data = pkl.load(open(fld_path, 'rb'))
    dataframe_accum = []

    for file in data.keys():
        dataframe = data[file]
        dataframe = pd.DataFrame.from_dict(dataframe)
        dataframe['filename'] = file
        dataframe_accum.append(dataframe.values)  
    
    df_array = np.concatenate(dataframe_accum)
    print("accumulated array shape: ", df_array.shape)
    df_array_crop = df_array[:,:n_attr]

    df_array_crop = df_array_crop.astype(float)

    # Plotting histogram for each attribute 
    for id in range(0, n_attr): 
        fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
        axs.hist(df_array_crop[:,id]) 
        # plt.savefig('./figs/attr-'+str(id)+'.png') 
        plt.clf()

    spread = df_array_crop.max(axis=0) - df_array_crop.min(axis=0)
    print("range: ", spread)
    stds = df_array_crop.std(axis=0)  
    means = df_array_crop.mean(axis=0) 
    
    return means, stds


# This module will process the predicted results for attribute classifier 
def process_results(fld_path, save_prefix, attach_mean_to_fname=False):  

    data = pkl.load(open(fld_path, 'rb'))
    # attr_list = ['male','smile','young','bald','eyeglasses','no-beard']  
    attr_list = list(data[list(data.keys())[0]].keys())
    print(attr_list)
    colors = ['green', 'red', 'violet', 'blue', 'maroon', 'orange', 'black'][:n_attr]
    mean_color = 'indigo'

    logits_means, logits_stds = compute_stats(fld_path)
    print("attr mean shape: ", logits_means.shape, " attr std shape: ", logits_stds.shape)
    print("attr mean: ", logits_means, " attr stds: ", logits_stds) 

    dirs_attr_stack = []
    fname_list = []

    # Iterating over all the directions 
    id_stop = 0
    for fn in tqdm(data.keys()):
        if (not fn == '.DS_Store'):
            
            fname_list.append(fn)
            dataframe = pd.DataFrame.from_dict(data[fn])
            dataframe['filename'] = fn
            # Adding the dataframe for a single image into the stack 
            dirs_attr_stack.append(dataframe.values) 
            id_stop += 1

    attr_list = dataframe.columns.values.tolist()
    print(attr_list)

    
    print("len of dirs attr stack: ", len(dirs_attr_stack), " shape of dirs attr stack0: ", dirs_attr_stack[0].shape)
    
    dirs_attr_array = [da[:,:n_attr] for da in dirs_attr_stack] 
    dirs_name_array = [da[:, -1] for da in dirs_attr_stack]

    delta_accum = []
    # Iterating over all the image directions, in this case we have each image has a separate direction
    for at_id in tqdm(range(0, len(dirs_attr_array))):

        delta = create_attr_ppl(dirs_attr_array[at_id], logits_stds, logits_means) 

        delta = np.array(delta)
        delta_accum.append(delta) 
        attr_save_dict = {}

        mean_attrs = np.mean(np.abs(delta[:, :n_attr]), axis=-1)
        attr_save_dict['mean'] = np.mean(mean_attrs)

        if attach_mean_to_fname:
            dst_path = os.path.join(save_prefix, dirs_name_array[at_id][0].replace('.mp4', '__' + str(np.around(mean_attrs.mean(), decimals=3)) + '.mp4'))
        else:
            dst_path = os.path.join(save_prefix, dirs_name_array[at_id][0])
        os.makedirs(dst_path, exist_ok=True)

        for attr in range(0, n_attr):
            if save_prefix is None:
                break

            attr_save_dict[attr_list[attr]] = np.mean(delta[:, attr])

            plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], delta[:, attr], label=attr_list[attr], linewidth=4, color=colors[attr]) 
            plt.plot([(1/seq_len), 1], [delta[-1, attr], delta[0, attr]], linewidth=4, color=colors[attr], linestyle = '--')
            img_dst_path = os.path.join(dst_path, attr_list[attr] + '.png')
            # plt.legend()
            # plt.savefig('./figs/attrs_bangs/result_attr_dir_ours'+fname_list[at_id]+'.png')  
            plt.savefig(img_dst_path)
            # print("--------- Saving result for -------------- :", fname_list[at_id])
            plt.clf()

        out_file = open(os.path.join(dst_path, "attrs.json"), "w")
        json.dump(attr_save_dict, out_file, indent=4)
        out_file.close()

        plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], mean_attrs, label="Mean of attributes", linewidth=4, color=mean_color)
        plt.plot([(1/seq_len), 1], [mean_attrs[-1], mean_attrs[0]], linewidth=4, color=mean_color, linestyle = '--')
        dst_path = os.path.join(save_prefix, dirs_name_array[at_id][0])
        os.makedirs(dst_path, exist_ok=True)
        dst_path = os.path.join(dst_path, 'attrs_mean.png')
        plt.savefig(dst_path)
        plt.clf()


    delta_accum = np.array(delta_accum)
    detla_accum_abs = np.abs(delta_accum)
    delta_accum_mean = detla_accum_abs.mean(axis=0)
    delta_accum_stds = detla_accum_abs.var(axis=0) ** 0.5
    # for attr in range(0, 6):
    #     plt.plot([(1/11)*i for i in range(0,11)], delta_accum_mean[:, attr], label=attr_list[attr])

    # plt.yscale('log')
    # plt.legend()
    # plt.title('attr-ppl-with-interpolation-random')
    # plt.savefig('./figs/graphs/result_rand_dir_accum_baseline'+str(at_id)+'.png')
    # plt.clf()       
    
    # print("delta_accum shape: ", delta_accum.shape)
    # print("dirs stack1 shape: ", dirs_attr_array[1].shape) 

    return delta_accum_mean, delta_accum_stds, attr_list


# This module will dump the results in a graph and numpy array 
def process_outputs(delta_accum_ours, delta_accum_baseline, delta_accum_ours_stds, delta_accum_baseline_stds, attr_list, type, save_prefix='.'):
    os.makedirs(save_prefix, exist_ok=True)
    colors = ['green', 'red', 'violet', 'blue', 'maroon', 'orange', 'black'][:n_attr]
    mean_delta_ours = delta_accum_ours.mean(axis=1)
    mean_delta_baseline = delta_accum_baseline.mean(axis=1)
    std_delta_ours = delta_accum_ours_stds.mean(axis=1)
    std_delta_baseline = delta_accum_baseline.mean(axis=1)

    # Means
    for attr in range(0, n_attr):
        plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], delta_accum_ours[:, attr], label=attr_list[attr].split('-')[-1].split('.')[0] + ' + HSR', color=colors[attr], alpha = 0.8)
        plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], delta_accum_baseline[:, attr], label=attr_list[attr].split('-')[-1].split('.')[0], linestyle='dashed', color=colors[attr], alpha = 0.8)
    
    plt.legend(prop={'size':14}, ncol=2)
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('Attribute Linearity Score', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')                         
    plt.savefig(os.path.join(save_prefix, 'test_all.pdf'))
    plt.clf()

    # Stds
    for attr in range(0, n_attr):
        plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], delta_accum_ours_stds[:, attr], label=attr_list[attr].split('-')[-1].split('.')[0] + ' + HSR', color=colors[attr], alpha = 0.8)
        plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], delta_accum_baseline_stds[:, attr], label=attr_list[attr].split('-')[-1].split('.')[0], linestyle='dashed', color=colors[attr], alpha = 0.8)
    
    plt.legend(prop={'size':14}, ncol=2)
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('ALS StdDev', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')                         
    plt.savefig(os.path.join(save_prefix, 'test_all_stds.pdf'))
    plt.clf() 


    # Means
    plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], mean_delta_ours, color='orange', label='SG2-ADA + HSR', linewidth=2)
    plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], mean_delta_baseline, color='blue', label='SG2-ADA ', linewidth=2)

    np.save('./figs/graph_data/random_init_ours.npy', delta_accum_ours)
    np.save('./figs/graph_data/random_init_baseline.npy', delta_accum_baseline)

    load_a = np.load('./figs/graph_data/random_init_ours.npy', allow_pickle=True)
    # print("are equal: ", load_a == delta_accum_ours)
    load_b = np.load('./figs/graph_data/random_init_baseline.npy', allow_pickle=True)
    # print("are equal: ", load_b == delta_accum_baseline)

    if (type == 'log'):
        plt.yscale('log')

    plt.legend(prop={'size':20})
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('Attribute Linearity Score', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18) 

    # plt.title('Mean PPL Attribute')  
    plt.savefig(os.path.join(save_prefix, 'test_mean.pdf'))
    plt.clf()



    # Stds
    plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], std_delta_ours, color='orange', label='SG2-ADA + HSR', linewidth=2)
    plt.plot([(1/seq_len)*i for i in range(seq_len,0,-1)], std_delta_baseline, color='blue', label='SG2-ADA ', linewidth=2)

    np.save('./figs/graph_data/random_init_ours_stds.npy', delta_accum_ours_stds)
    np.save('./figs/graph_data/random_init_baseline_stds.npy', delta_accum_baseline_stds)

    load_a = np.load('./figs/graph_data/random_init_ours_stds.npy', allow_pickle=True)
    # print("are equal: ", load_a == delta_accum_ours)
    load_b = np.load('./figs/graph_data/random_init_baseline_stds.npy', allow_pickle=True)
    # print("are equal: ", load_b == delta_accum_baseline)

    if (type == 'log'):
        plt.yscale('log')

    plt.legend(prop={'size':20})
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('ALS StdDev', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')  
    plt.savefig(os.path.join(save_prefix, 'test_std.pdf'))
    plt.clf()


# This module will dump the results in a graph and numpy array
def process_outputs_single(delta_accum_ours, delta_accum_ours_stds, attr_list, type, save_prefix='.'):
    os.makedirs(save_prefix, exist_ok=True)
    colors = ['green', 'red', 'violet', 'blue', 'maroon', 'orange', 'black'][:n_attr]
    mean_delta_ours = delta_accum_ours.mean(axis=1)
    std_delta_ours = delta_accum_ours_stds.mean(axis=1)

    # Means
    for attr in range(0, n_attr):
        plt.plot([(1 / seq_len) * i for i in range(seq_len, 0, -1)], delta_accum_ours[:, attr],
                 label=attr_list[attr].split('-')[-1].split('.')[0] + ' + HSR', color=colors[attr], alpha=0.8)

    plt.legend(prop={'size': 14}, ncol=2)
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('Attribute Linearity Score', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')
    plt.savefig(os.path.join(save_prefix, 'test_all.pdf'))
    plt.clf()

    # Stds
    for attr in range(0, n_attr):
        plt.plot([(1 / seq_len) * i for i in range(seq_len, 0, -1)], delta_accum_ours_stds[:, attr],
                 label=attr_list[attr].split('-')[-1].split('.')[0] + ' + HSR', color=colors[attr], alpha=0.8)

    plt.legend(prop={'size': 14}, ncol=2)
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('ALS StdDev', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')
    plt.savefig(os.path.join(save_prefix, 'test_all_stds.pdf'))
    plt.clf()

    # Means
    plt.plot([(1 / seq_len) * i for i in range(seq_len, 0, -1)], mean_delta_ours, color='orange', label='SG2-ADA + HSR',
             linewidth=2)

    np.save('./figs/graph_data/random_init_ours.npy', delta_accum_ours)

    if (type == 'log'):
        plt.yscale('log')

    plt.legend(prop={'size': 20})
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('Attribute Linearity Score', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')
    plt.savefig(os.path.join(save_prefix, 'test_mean.pdf'))
    plt.clf()

    # Stds
    plt.plot([(1 / seq_len) * i for i in range(seq_len, 0, -1)], std_delta_ours, color='orange', label='SG2-ADA + HSR',
             linewidth=2)

    np.save('./figs/graph_data/random_init_ours_stds.npy', delta_accum_ours_stds)

    if (type == 'log'):
        plt.yscale('log')

    plt.legend(prop={'size': 20})
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('ALS StdDev', fontsize=18)
    plt.xlabel('Interpolation Variable $\it{t}$', fontsize=18)

    # plt.title('Mean PPL Attribute')
    plt.savefig(os.path.join(save_prefix, 'test_std.pdf'))
    plt.clf()


# This module will compute the area under the graph for each of the attribute delta between the prediction and the interpolated value 
def compute_metrics(delta_accum_ours, delta_accum_baseline, delta_accum_ours_stds, delta_accum_baseline_stds):

    area_delta_ours = np.trapz(delta_accum_ours, dx=0.1, axis=0)
    area_delta_baseline = np.trapz(delta_accum_baseline, dx=0.1, axis=0)

    area_slim_delta_ours = np.sum(np.square(delta_accum_ours), axis=0) 
    area_slim_delta_baseline = np.sum(np.square(delta_accum_baseline), axis=0)

    print("area_delta_ours:") 
    # [print(round(area_delta_ours[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta_ours[id],3)) for id in range(0,n_attr)]   

    print("area_delta_baseline:")  
    # [print(round(area_delta_baseline[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta_baseline[id],3)) for id in range(0,n_attr)]    

    area_mean_ours = area_delta_ours.mean()
    area_mean_baseline = area_delta_baseline.mean()

    slim_mean_our = area_slim_delta_ours.mean() 
    slim_mean_baseline = area_slim_delta_baseline.mean()

    # print("area mean ours: ", round(area_mean_ours, 3))
    print("area mean our:", slim_mean_our)
    # print("area mean baseline: ", round(area_mean_baseline, 3))
    print("area mean baseline:", slim_mean_baseline)
    

    area_delta_ours_stds = np.trapz(delta_accum_ours_stds, dx=0.1, axis=0)
    area_delta_baseline_stds = np.trapz(delta_accum_baseline_stds, dx=0.1, axis=0)

    area_slim_delta_ours_stds = np.sum(np.square(delta_accum_ours_stds), axis=0)
    area_slim_delta_baseline_stds = np.sum(np.square(delta_accum_baseline_stds), axis=0)

    print("area_delta_ours_stds:") 
    # [print(round(area_delta_ours[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta_ours_stds[id],3)) for id in range(0,n_attr)]

    print("area_delta_baseline_stds:")
    # [print(round(area_delta_baseline[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta_baseline_stds[id],3)) for id in range(0,n_attr)]
      
    area_stds_ours = area_delta_ours_stds.mean()
    area_stds_baseline = area_delta_baseline_stds.mean()

    slim_stds_our = area_slim_delta_ours_stds.mean() 
    slim_stds_baseline = area_slim_delta_baseline_stds.mean()

    # print("area mean ours: ", round(area_mean_ours, 3))
    print("area stds our:", slim_stds_our)
    # print("area mean baseline: ", round(area_mean_baseline, 3))
    print("area stds baseline:", slim_stds_baseline)


# This module will compute the area under the graph for each of the attribute delta between the prediction and the interpolated value
def compute_metrics_individual(delta_accum, delta_accum_stds):
    area_delta_ours = np.trapz(delta_accum, dx=0.1, axis=0)

    area_slim_delta = np.sum(np.square(delta_accum), axis=0)

    print("area_delta:")
    # [print(round(area_delta_ours[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta[id], 3)) for id in range(0, n_attr)]

    slim_mean = area_slim_delta.mean()

    # print("area mean ours: ", round(area_mean_ours, 3))
    print("area mean our:", slim_mean)

    area_delta_stds = np.trapz(delta_accum_stds, dx=0.1, axis=0)

    area_slim_delta_stds = np.sum(np.square(delta_accum_stds), axis=0)

    print("area_delta_stds:")
    # [print(round(area_delta_ours[id],3)) for id in range(0,6)]
    [print(round(area_slim_delta_stds[id], 3)) for id in range(0, n_attr)]

    area_stds_ours = area_delta_stds.mean()

    slim_stds_our = area_slim_delta_stds.mean()

    # print("area mean ours: ", round(area_mean_ours, 3))
    print("area stds our:", slim_stds_our)


def run_main():
    pkl_path = 'path/to/pkl'
    save_path_prefix = 'output/path'
    delta_accum, delta_accum_stds, attr_list = process_results(pkl_path, save_prefix=save_path_prefix, attach_mean_to_fname=True)

    save_path_prefix_graphs = 'output/path/mean_graphs'
    process_outputs_single(delta_accum, delta_accum_stds, attr_list, 'no-log', save_prefix=save_path_prefix_graphs)
    compute_metrics_individual(delta_accum, delta_accum_stds)

   
if __name__ == "__main__":
    run_main()

