import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
plt.rcParams["font.size"] = 17
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams["figure.dpi"]=120
plt.rcParams["legend.fancybox"] = False # 丸角
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in


if __name__ == "__main__":

    method_dic = {}
    color_dic={}
    cmap = plt.get_cmap("tab10")
    params = "n1500_l600_alpha03_beta05_initial05488135039273248"#"_n500_l100_alpha03_beta05"
    if not os.path.exists('./figures/'+params):
        os.makedirs('./figures/'+params)
    save_dir_name = './figures/'+params + '/'


    i=0
    for name in ["GD","RNM","RSRNM_seed1_d100", "RSRNM_seed1_d200", "RSRNM_seed1_d400"]:
        method_dic[name] = (np.loadtxt('./results/' + params +'/' + name + '.txt')).T
        color_dic[name]  = cmap(i)
        i += 1
    print(color_dic)

    frame_alpha = 0.7


    key = {"iteration":0, "time":1, "function_value":2, "grad_norm":3}

    interval = 1
    GD_interval = 6

    #iter vs function_value
    x_label = "iteration"
    y_label = "function_value"
    for name, data in method_dic.items():
        print(name, data.shape)
        if (name[:5]!="RSRNM"):
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r""+name, color = color_dic[name], zorder=2.1)
        else:
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r"RS-RNM($s=$"+name[13:]+")", color = color_dic[name], zorder=2.0)
    plt.xlabel(r'iterations')
    plt.ylabel(r'$f(w)$')
    plt.xlim(0,700)
    #plt.ylim(0,0.2)
    plt.yscale('log')
    plt.legend(fontsize=14.5, framealpha=frame_alpha)
    plt.tight_layout()
    plt.savefig(save_dir_name+x_label[:4]+"_"+y_label[:4]+".png")
    #plt.close()
    plt.show()


    #iter vs grad_norm
    x_label = "iteration"
    y_label = "grad_norm"
    for name, data in method_dic.items():
        print(name, data.shape)
        if (name[:5]=="RSRNM"):
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r"RS-RNM($s=$"+name[13:]+")", color = color_dic[name], zorder=2.1)
        else:
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r""+name, color = color_dic[name], zorder=2.2)
    plt.xlabel(r'iterations')
    plt.ylabel(r'$\|\nabla f(w)\|$')
    plt.xlim(0,700)
    plt.ylim(1e-5,6.0)
    plt.yscale('log')
    plt.legend(fontsize=14.5, framealpha=frame_alpha, loc='lower right', bbox_to_anchor=(1.0, 0.1))
    plt.tight_layout()
    plt.savefig(save_dir_name+x_label[:4]+"_"+y_label[:4]+".png")
    #plt.close()
    plt.show()


    #time vs function_value
    x_label = "time"
    y_label = "function_value"
    for name, data in method_dic.items():
        print(name, data.shape)
        if (name[:5]!="RSRNM"):
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r""+name, color = color_dic[name], zorder=2.1)
        else:
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r"RS-RNM($s=$"+name[13:]+")", color = color_dic[name], zorder=2.0)
    plt.xlabel(r'time(s)')
    plt.ylabel(r'$f(w)$')
    plt.xlim(0,70)
    #plt.ylim(0,0.2)
    plt.yscale('log')
    plt.legend(fontsize=14.5, framealpha=frame_alpha)
    plt.tight_layout()
    plt.savefig(save_dir_name+x_label[:4]+"_"+y_label[:4]+".png")
    #plt.close()
    plt.show()


    #time vs grad_norm
    x_label = "time"
    y_label = "grad_norm"
    for name, data in method_dic.items():
        print(name, data.shape)
        if (name[:5]=="RSRNM"):
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r"RS-RNM($s=$"+name[13:]+")", color = color_dic[name], zorder=2.1)
        elif (name=="GD"):
            plt.plot(data[key[x_label]][::GD_interval], data[key[y_label]][::GD_interval], label=r""+name, color = color_dic[name], zorder=2.2)
        else:
            plt.plot(data[key[x_label]][::interval], data[key[y_label]][::interval], label=r""+name, color = color_dic[name], zorder=2.2)
    plt.xlabel(r'time(s)')
    plt.ylabel(r'$\|\nabla f(w)\|$')
    plt.yscale('log')
    plt.xlim(0,70)
    plt.ylim(1e-5,6.0)
    plt.legend(fontsize=14.5, framealpha=frame_alpha)#, loc='lower right', bbox_to_anchor=(1.0, 0.1))
    plt.tight_layout()
    plt.savefig(save_dir_name+x_label[:4]+"_"+y_label[:4]+".png")
    #plt.close()
    plt.show()
