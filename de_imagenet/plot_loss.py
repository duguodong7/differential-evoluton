from matplotlib import ticker
import matplotlib.pyplot as plt
import random
import pandas as pd


def plot(file):
    df = pd.read_csv(file)
    index = 0
    pop_size=10
    df=df[index:]
    val_acc=[]
    for i in range(len(df["eval_top1"])):
        if(i%2==0):
    #         print(i)
            val_acc.append(eval(df["eval_top1"][i+index]))
    val_acc=np.array(val_acc)
    val_loss=[]
    for i in range(len(df["eval_eval_loss"])):
        if(i%2==0):
    #         print(i)
            val_loss.append(eval(df["eval_eval_loss"][i+index]))
    val_loss=np.array(val_loss)
    scaling1=1
    new_pop_acc = []
    old_pop_acc=[]
    for i in range(len(val_acc)):
        for j in range(len(val_acc[1])):
            if i == 0:
                old_pop_acc.append([i,val_acc[i][j]])
            else:
                new_pop_acc.append([i,val_acc[i][j]])
    old_pop_acc=np.array(old_pop_acc)
    new_pop_acc=np.array(new_pop_acc)
    new_pop_loss = []
    old_pop_loss=[]
    for i in range(len(val_loss)):
        for j in range(len(val_loss[1])):
            if i == 0:
                old_pop_loss.append([i,val_loss[i][j]])
            else:
                new_pop_loss.append([i,val_loss[i][j]])
    old_pop_loss=np.array(old_pop_loss)
    new_pop_loss=np.array(new_pop_loss)
    adam_acc=val_acc[0]
    adam_loss=val_loss[0]
    optimizer_change_point=10
    fig,ax = plt.subplots(1,1,figsize = (20,15))
    ax.plot([x*scaling1 if x<=0 else x for x in range(-optimizer_change_point,0)],adam_loss[:optimizer_change_point],color='#0557FA',label="DE_Acc_Vali",alpha=1)
    ax.scatter(x=new_pop_loss[:,0],y=new_pop_loss[:,1],marker='x',color='#F35858',label="DE")
    ax.scatter(x=old_pop_loss[:,0],y=old_pop_loss[:,1],marker='x',color='#007FFF',label="DE")
    ax.tick_params(labelsize = 50)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(pad = 25)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for tick in ax.xaxis.get_ticklines():
        tick.set_markersize(10)
        tick.set_markeredgewidth(0.5)
    for tick in ax.yaxis.get_ticklines():
        tick.set_markersize(10)
        tick.set_markeredgewidth(0.5)
    left, bottom, width, height = [0.57, 0.53, 0.328, 0.34]
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot([x*scaling1 if x<=0 else x for x in range(-optimizer_change_point,0)],adam_acc[:optimizer_change_point],color='#0557FA',label="DE_Acc_Vali",alpha=1)
    ax1.scatter(x=new_pop_acc[:,0],y=new_pop_acc[:,1],marker='x',color='#F35858',label="DE")
    ax1.scatter(x=old_pop_acc[:,0],y=old_pop_acc[:,1],marker='x',color='#007FFF',label="DE")
    ax1.tick_params(labelsize = 40)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.tick_params(pad = 15)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    for tick in ax1.xaxis.get_ticklines():
        tick.set_markersize(10)
        tick.set_markeredgewidth(0.5)
    for tick in ax1.yaxis.get_ticklines():
        tick.set_markersize(10)
        tick.set_markeredgewidth(0.5)
    plt.savefig('2.svg')
