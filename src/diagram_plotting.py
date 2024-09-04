import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import vonmises
from scipy.stats import norm
import matplotlib.ticker as mtick
import os
from wtconversion import *
from experimental_phase_fields import *

#code to check where first nan is in failed df


'''
for score,label in zip(scores,labels):
    print(f'score:{score}')
    x=[]
    labelsb=[]
    for kappa in df_nulled['kappa'].unique():
        df_s=df_nulled[df_nulled['kappa']==kappa]
        df_s=df_s[df_s.columns[6+score:129+score][::4]]
        x.append(df_s.max(axis=1))
        labelsb.append(len(x[-1]))
        print(len(x))
        for i in x:
            print(len(i))
    labelsa=df_nulled['kappa'].unique()
    boxlabels=[f"{x}(Data size = {str(y)}" for x,y in zip(labelsa,labelsb)]
    plt.boxplot(x,labels=boxlabels)
    plt.xlabel('Kappa')
    plt.ylabel(label)
    plt.title('Maximum score distribution for nulled simulations')
    plt.savefig(f'../simulation/20_02/max_score_dist_{label}')
    plt.clf()
    '''
#code for plotting line
'''
#define score metric (0=d(nu,u), 1=true second moment, 2=d(closest,u)
df_g=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
#df_g=df
for score,label in zip(range(4),
                       ['d(mu,u)','true second moment','d(closest u)',
                        'purity']):
    fig,axs=plt.subplots(1,3)
    fig.suptitle(f'Score type is {label}')

    for m,kappa in enumerate(df_g['Kappa'].unique()):
        ax=axs[m]
        ax.title.set_text(f'Kappa = {kappa}')
        for batch_size in df_g['Batch size'].unique():
            for prior_size in df_g['Prior size'].unique():
                #select sub df
                df_red=df_g[
                    (df_g['Kappa']==kappa)
                    &(df_g['Batch size']==batch_size)
                    &(df_g['Prior size']==prior_size)]
                
                #get first column and last column
                found=False
                for n,i in enumerate(df_red.columns):
                    if label in i:
                        if not found:
                            found=True
                            first_index=n
                        last_index=n

                print(first_index)
                print(df_red.columns[first_index])
                print(last_index)
                print(df_red.columns[last_index])
                #get score 
                columns=df_red.columns[first_index:last_index+1][::4]
                scores=[]
                scores_std=[]
                for n,i in enumerate(columns):
                    print('xxxx')
                    print(i)
                    result=df_red[i].to_numpy()
                    if score ==3:
                        result*=100
                    scores.append(np.mean(result))
                    scores_std.append(np.std(result))
                print(len(columns))
                print(len(scores))
                print(len(scores_std))
                label=f'Batch size={batch_size},Prior size={prior_size}'
                ax.errorbar(
                    range(len(columns)),scores,yerr=scores_std,label=label,
                    capsize=5)
    plt.legend()
    plt.show()
    '''
#code for plotting violins
'''
df=A value is trying to be set on a copy of a slice from a DataFrame.
pd.read_csv('../simulation/1/MgAlCu.csv')
df_g=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
#for score,label in zip(range(4),
                       #['d(mu,u)','true second moment','d(closest u)',
                        #'purity']):
for score,label in zip([3],['purity']):

    for m,kappa in enumerate(df_g['Kappa'].unique()):
        for batch_size in df_g['Batch size'].unique():
            for prior_size in df_g['Prior size'].unique():
                df_red=df_g[
                    (df_g['Kappa']==kappa)
                    &(df_g['Batch size']==batch_size)
                    &(df_g['Prior size']==prior_size)]
                
                #get first column and last column
                found=False
                for n,i in enumerate(df_red.columns):
                    if label in i:
                        if not found:
                            found=True
                            first_index=n
                        last_index=n

                print(first_index)
                print(df_red.columns[first_index])
                print(last_index)
                print(df_red.columns[last_index])
                #get score 
                data=df_red[df_red.columns[first_index:last_index+1][::4]].to_numpy()[:,0:7]
A value is trying to be set on a copy of a slice from a DataFrame.
                print(data.shape)
                fig,ax=plt.subplots(1,1)
                ax.title.set_text(f'Kappa={kappa}, Batch size={batch_size},Pri'
                                  +f'or size={prior_size}, Score type={label}')
                #select sub df
                ax.violinplot(data)
                #plt.show()
                plt.savefig('../output_sim/violin/MgAlCu/k='
                            +f'{kappa}_b={batch_size}_p={prior_size}')
                plt.close()
                '''
#code for getting average batch for x purity
'''
df=pd.read_csv('../simulation/1/LiAlBO.csv')
df=df[
    (df['Kappa']==100)
    &(df['Prior size']==3)
    &(df['Batch size']==3)]
                #get first column and last column
found=False
for n,i in enumerate(df.columns):
    if 'purity' in i:
        if not found:
            found=True
            first_index=n
        last_index=n
#get score 
data=df[df.columns[first_index:last_index+1][::4]].to_numpy()[:,:10]
data[data==np.nan]=0
print(data.shape)
threshold=0.95
x=np.argmax(data>threshold,axis=1)+1
print(np.mean(x))
print(np.std(x))
'''

'''
                    label=(f'Kappa:{kappa},n:{b_s},Success rate:'


                           + f'{len(df_n)/len(df):.2%}'),capsize=4)
            #df_not_found=df[df['End type']=='Terminated successfully']
            #df_found=df[df['End type']=='Found']
            #df_nulled=df[df['End type']=='Nulled']
            #plot full plot
            #something
            score_a=[]
            score_a_std=[]
            for i in df_n.columns[6+score:129+score][::4]:
                print(i)
                result=df_n[i].to_numpy()
                if score ==3:
                    result*=100
                score_a.append(np.mean(result))
                score_a_std.append(np.std(result))
            ax.errorbar(
                range(31),score_a,yerr=score_a_std,
                label=(f'Kappa:{kappa},n:{b_s},Success rate:'
                       + f'{len(df_n)/len(df):.2%}'),capsize=4)
    if score == 3:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel('Batch number')
    plt.ylabel(label)
    plt.legend()
    #plt.show()
    plt.savefig(f'../simulation/20_02/{label}vsBatch_U={u}.png')
        '''


'''
for kappa in df_nulled['kappa'].unique():
    print(f'kappa={kappa}')
    df=df_nulled[df_nulled['kappa']==kappa]

    for i in df.columns[6:39][::4]:
        print(f'batch {i} contains nan? {np.any(np.isnan(df[i].unique()))}')
        print(df[i].unique())
#print(df_nulled.iloc[:,36].unique())
#code for printing amount of different end types
df_a=pd.read_csv('../simulation/2001_1.csv')
for i in df_a['kappa'].unique():
    print(i)
    df=df_a[df_a['kappa']==i]
    df_not_found=df[df['End type']=='Terminated successfully']
    df_found=df[df['End type']=='Found']
    df_nulled=df[df['End type']=='Nulled']
    print(df_nulled['true second moment for batch28'])
    print(f'Found: {len(df_found)}, Not found: {len(df_not_found)}, Nulled: '
          +f'{len(df_nulled)}')
'''
#other code for gauging kappa
'''
diffs=[]
kappas=[100,1000,10000]
x=np.arange(0,1.00,0.1)
y=1-x
mu=np.arctan(x/y)
r=np.sqrt(x**2+y**2)
looking=True
thetas=[]
for xs,m,rs in zip(x,mu,r):
    plt.title(f'theta={m},x={xs}')
    for n,kappa in enumerate(kappas):
        theta_t = vonmises.rvs(kappa, loc=m,size=500000)
        #plt.hist(theta_t,bins=100,alpha=0.5,label=f'k={kappa}')

        theta_t=theta_t[(theta_t<np.pi/2) & (theta_t>0)][:200000]
        x_new=rs*np.sin(theta_t)
        plt.hist(x_new,bins=100,alpha=0.5,label=f'k={kappa}')
        #plt.hist(theta_t,bins=100,alpha=0.5,label=f'k={kappa},trunked',density=True)
        thetas.append(theta_t)
    plt.legend()
    plt.show()
    plt.savefig(f'../output_sim/kappa_gauge/x={xs}.png')
    '''
#code for gauging kappa
'''
kappas=[100,1000,10000]
std_all=[]
sq_e_all=[]
for n,kappa in enumerate(kappas):
    diffs=[]
    x=np.arange(0.001,1,0.01)
    y=1-x
    r=np.sqrt(x**2+y**2)
    mu=np.arctan(x/y)
    looking=True
    thetas=[]
    for m in mu:
        theta_t = vonmises.rvs(kappa, loc=m,size=500000)
        theta_t=theta_t[(theta_t<np.pi/2) & (theta_t>0)][:200000]
        thetas.append(theta_t)
    stdx=[]
    stdy=[]
    for theta,co,rs in zip(thetas,x,r):
        theta=np.array(theta)
        x_new=rs*np.sin(theta)
        y_new=np.cos(theta)
        x_diff=x_new-co
        y_diff=y_new-1+co
        diffs+=list(x_diff)
        stdy.append(np.std(y_new))
    diffs=100*np.array(diffs)
    #std_all.append((1/len(diffs))*np.sqrt(np.sum(diffs**2)))
    std_all.append(np.std(diffs))
    sq_e_all.append(np.mean(diffs**2))

    #ax[0][n].errorbar(x,[0 for i in x],yerr=stdx)
    #ax[1][n].errorbar(y,[0 for i in y],yerr=stdy)
    '''
#plt.plot(kappas,sq_e_all,'+')
#plt.xlabel('Kappa')
#plt.ylabel('Simulated standard deviation on the input relative molar ratios')
#plt.title('Aproximate relationship between kappa and STD')
#plt.show()
'''
for k,e in zip(kappas,np.array(sq_e_all)**0.5):
    print(k,":",e)
    '''
#code for plotting score vs batch number
#success rate gives % of time probability was nulled
'''
df_a=pd.read_csv('../simulation/2001_1.csv')
for i in df.columns:
    print(i)
print(df_a['Unknown'].unique())
#define score metric (0=d(nu,u), 1=true second moment, 2=d(closest,u)
for u in df_a['Unknown'].unique():
    df_u=df_a[df_a['Unknown']==u]
    df_n=df_u[df_u['End type']!='Nulled']
    for score,label in zip(range(4),
                           ['d(mu,u)','True second moment','d(closest u)',
                            'Percentage purity']):
        ress=[]
        fig,ax=plt.subplots(1,1)
     Gigabit Ethernet         for kappa in df_a['kappa'].unique():
            scores=[]
            labels=[]
            print(f'kappa={kappa}')
            for b_s in df_a['Batch size'].unique():
                df=df_a[(df_a['kappa']==kappa)&(df_a['Batch size']==b_s)]
                #df_not_found=df[df['End type']=='Terminated successfully']
                #df_found=df[df['End type']=='Found']
                #df_nulled=df[df['End type']=='Nulled']
                #plot full plot
                #something
                df_n=df[df['End type']=='Found']
                score_a=[]
                score_a_std=[]
                for i in df_n.columns[6+score:129+score][::4]:
                    print(i)
                    result=df_n[i].to_numpy()
                    if score ==3:
                        result*=100
                    score_a.append(np.mean(result))
                    score_a_std.append(np.std(result))
                ax.errorbar(
                    range(31),score_a,yerr=score_a_std,
                    label=(f'Kappa:{kappa},n:{b_s},Success rate:'
                           + f'{len(df_n)/len(df):.2%}'),capsize=4)
        if score == 3:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel('Batch number')
        plt.ylabel(label)
        plt.legend()
        #plt.show()
        plt.savefig(f'../simulation/20_02/{label}vsBatch_U={u}.png')
        '''

        #expected score
'''
        res=[kappa,b_s]
        df_found=df[df['End type']=='Found']
        found=0
        score_a=[]
        for n,i in enumerate(df_found.columns[6:99][::-3]):
            print(i)
            df_succ=df_found[df_found[i]==0]
            score_a+=[n]*(len(df_succ)-found)
            found+=len(df_succ)
            df_found[df_found[i]==0]['Score']=30-n
        res.append(np.mean(score_a))
        res.append(np.std(score_a))
        res.append(round(len(df_found)/len(df),2))
        ress.append(res)
df_res=pd.DataFrame(
    columns=['Kappa','Batch size','Mean steps','STD','Fraction success'],
    data=ress)
print(df_res.Automated Phase Isolationto_string(index=False))
plt.legend()
plt.show()
'''


#df=pd.read_csv('../simulation/2/MgAlCu.csv')
#print(len(df)/(27*len(df['Unknown'].unique())))
#df=pd.read_csv('../simulation/2/LiAlBO.csv')
#print(len(df)/(27*len(df['Unknown'].unique())))
#df=pd.read_csv('../simulation/2/MgBOF.csv')
#print(len(df)/(27*len(df['Unknown'].unique())))

#distribution after nth batch
'''
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
phase_fields_names=["Mg$^{2+}$-B$^{3+}$-O$^{2-}$-F$^{1-}$","Mg-Al-Cu",
              "Li$^{1+}$-Al$^{3+}$-B$^{3+}$-O$^{2-}$"]
plt.rcParams['figure.figsize'] = (6.50127*0.50),1.8
plt.rcParams.update({'font.size': 8})
for field,name in zip(phase_fields,phase_fields_names):
    df=pd.read_csv('../simulation/reduced/'+field+'.csv')
    df=df[(df['End type']!='Prior was null')]
    fig,ax=plt.subplots()
    frames=[]
    for m,std in enumerate([0.1,0.02]):
    #for m,std in enumerate([0.1,0.02]):
        dfa=pd.DataFrame()
Automated Phase Isolation        n=0
        score=[]
        #for batch_size in [1,3,5]:
        for batch_size in [1,3,5]:
            df_red=df[(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            print('field:',field,'std:',std,'b:',batch_size)
            print('score = ',df_red['Score 3'].mean())
            frames.append(df_red)
    df=pd.concat(frames)
    df['Experimental $\sigma$']=df['Experimental std'].astype('category')
    df['Batch size']=df['Batch size'].astype('category')
    #df['Prior size']=df['Prior size'].astype('category')
    #df['Prior size | Batch size']=(df['Prior size'].astype(str) + " | " 
    #                               +df['Batch size'].astype(str))
    #print(df['Prior size | Batch size'])
    df['Score 3']*=100
    sns.boxplot(
        df,x='Score 3',hue='Experimental $\sigma$',
        y='Batch size',
            showfliers=False)
    plt.xlabel("Purity score /%")
    plt.tight_layout(pad=0)
    plt.legend(loc='lower left',title="$\sigma$")
    plt.savefig("../../paper/3dist_"+field+".png",dpi=1000)
    plt.show()
#plt.show()
#plotting expecte num batches
'''

#data refiner
#phase_fields=["MgBOF"]#,"MgAlCu","LiAlBO"]
phase_fields=["LiAlBO"]
f="0"
for field in phase_fields:
    #df=pd.read_csv('../simulation/data/compare_batches/'+field+'_'+str(f)+'.csv')
    df=pd.read_csv('../simulation/data/testing/'+field+"_"+str(f)+'.csv')
    for i in df.columns:
        print(i)
    stds=df['Experimental std'].unique()
    pes=df['Predicted error'].unique()
    bs=df['Batch size'].unique()
    us=df['Unknown'].unique()
    x=len(pes)*len(stds)*len(us)*len(bs)
    repeats=len(df)/x
    print(df.head())
    print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')
    #df['Predicted error'].astype('string').hist(bins=100)
    #plt.show()
    #df=df.iloc[300*len(df['Unknown'].unique())]
    #df=df[df['Prior size']==1]
    df_new=pd.DataFrame()
    df_new['Batch size']=df['Batch size']
    df_new['End type']=df['End type']
    df_new['Experimental std']=df['Experimental std']
    df_new['Predicted error']=df['Predicted error']
    df_new['Unknown']=df['Unknown']
    df_new['Initial distance']=df['d(closest,u) for batch0']
    df_new['Score 0']=df['purity for batch0']
    df_new['Score 1']=df['purity for batch1']
    df_new['Score 2']=df['purity for batch2']
    df_new['Score 3']=df['purity for batch3']
    df_new['Score 4']=df['purity for batch4']
    df_new['Score 5']=df['purity for batch5']
    df_new['Score 6']=df['purity for batch6']
    df_new['Score 7']=df['purity for batch7']
    df_new['Score 8']=df['purity for batch8']
    df_new['Score 9']=df['purity for batch9']
    df_new['Score 10']=df['purity for batch10']
    df_new['Score 11']=df['purity for batch11']
    df_new['Score 12']=df['purity for batch12']
    df_new['Score 13']=df['purity for batch13']
    df_new['Score 14']=df['purity for batch14']
    df_new['Score 15']=df['purity for batch15']
    #df_new.loc[(df_new['Score 14']!=df_new['Score 5'])&
    #       (df_new['End type']=='Nulled'),'End type']='Nulled late'
    #print(df_new['End type'].unique())
    df_new.to_csv(
        '../simulation/data/compare_batches/refined_'+field+"_"+str(f)+".csv",
        index=False)
sys.exit()

#score vs num_samples
'''
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
plt.rcParams['figure.figsize'] = (6.50127*0.50),3
plt.rcParams.update({'font.size': 8})
for field in phase_fields:
    df=pd.read_csv('../simulation/reduced/expanded/'+field+'.csv')
    df=df[(df['End type']=='Terminated success')|(df['End type']=='Found')]
    print(df['End type'].unique())
    print(df.columns)
uolvpn.liverpool.ac.uk/unmanaged    fig,ax=plt.subplots()
    for m,std in enumerate([0.1,0.02]):
        for batch_size in [1,3,5]:
            df_red=df[(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            xs=[]
            ys=[]
            ylows=[]
            yhighs=[]
            for batch_number in range(1,16):
                n=batch_size*batch_number
                if n<=15:
                    xs.append(n)
                    y=np.median(df_red['Score '+str(batch_number)])
                    ys.append(y*100)
                    ylows.append(
                        y-np.percentile(df_red['Score '+str(batch_number)],2))
                    yhighs.append(
                        np.percentile(df_red['Score '+str(batch_number)],98)-y)
            label="Batch size="+str(batch_size)
            plt.errorbar(xs,ys,yerr=[ylows,yhighs],label=label,capsize=8)
        plt.ylabel('Maximum purity of new crystalline phase/ %')
        plt.xlabel('Total number of samples')
        plt.legend(loc='lower right')
        plt.tight_layout(pad=0)
        plt.savefig('../../paper/batcheffect_'+field+str(std)+'.svg')
        plt.show()
'''
#score vs num_samples
'''
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
phase_fieldsf=["Mg^{2+}- B^{3+}- O^{2-}- F^-$","Mg-Al-Cu","Li$^{1+}$- Al$^{3+}$- B$^{3+}$- O$^{2-}$"]
#plt.rcParams['figure.figsize'] = (6.50127),2.5
plt.rcParams.update({'font.size': 8})
#cn=-1
#c=['darkred','lightcoral','darkgreen','lightgreen','darkblue','cornflowerblue']
width=6.49
height=3
dpi=600
fig,ax=plt.subplots(1,3,figsize=(width,height),dpi=dpi)
for p,field in enumerate(phase_fields):
    #cn+=1
    df=pd.read_csv('../simulation/data/'+field+'.csv')
    df=df[(df['End type']=='Terminated success')|(df['End type']=='Found')]
    print(df['End type'].unique())
    print(df.columns)
    for m,std in enumerate([0.1,0.02]):
        for batch_size in [1]:#,3,5]:
            df_red=df[(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            xs=[]
            ys=[]
            ylows=[]
            yhighs=[]
            for batch_number in range(1,16):
                n=batch_size*batch_number
                if n<=15:
                    xs.append(n)
                    y=np.median(df_red['Score '+str(batch_number)])
                    print(field,std,y)
                    ys.append(y*100)
                    ylows.append(
                        y-np.percentile(df_red['Score '+str(batch_number)],2))
                    yhighs.append(
                        np.percentile(df_red['Score '+str(batch_number)],98)-y)
            label="$\sigma$="+str(std)
            ax[p].errorbar(
                xs,ys,yerr=[ylows,yhighs],label=label,capsize=4,linewidth=1,capthick=1)
            ax[p].set_ylabel('Purity Score')
            ax[p].set_ylim([60,105])
            ax[p].set_xlabel('Total number of samples')
            ax[p].set_xticks(range(0,16,3))
            ax[p].set_yticks(range(60,101,5))
            ax[p].legend(loc='lower right')
            #ax[p].set_title(phase_fieldsf[p]+" Phase Field")
#plt.tight_layout(pad=0)
plt.tight_layout()
plt.savefig('../simulation/graphs/b=1_'+field+'.pdf')
plt.show()
'''
#Score vs number of samples gpt-test
'''
# Define phase fields and their formatted names
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
phase_fieldsf = ["Mg^{2+}- B^{3+}- O^{2-}- F^-", "Mg-Al-Cu", "Li$^{1+}$- Al$^{3+}$- B$^{3+}$- O$^{2-}$"]

# Configure plot settings
plt.rcParams.update({'font.size': 8})
width = 6.49
height = 3
save_dpi = 600
screen_dpi=100
fig, ax = plt.subplots(1, 3, figsize=(width, height), dpi=screen_dpi)

# Define the std and batch sizes to consider
selected_stds = [0.1, 0.02]
selected_batch_sizes = [1]

# Iterate over phase fields
for p, field in enumerate(phase_fields):
    df = pd.read_csv('../simulation/data/' + field + '.csv')
    df = df[(df['End type'] == 'Terminated success') | (df['End type'] == 'Found')]
    print(df['End type'].unique())
    print(df.columns)

    # Filter the DataFrame to include only selected std and batch sizes
    df = df[df['Experimental std'].isin(selected_stds) & df['Batch size'].isin(selected_batch_sizes)]

    # Set the multi-index using 'Experimental std' and 'Batch size'
    df.set_index(['Experimental std', 'Batch size'], inplace=True)

    # Group by the multi-index and iterate over the groups
    for (std, batch_size), df_group in df.groupby(level=['Experimental std', 'Batch size']):
        xs = []
        ys = []
        ylows = []
        yhighs = []

        # Calculate statistics for each batch number
        for batch_number in range(1, 16):
            n = batch_size * batch_number
            if n <= 15:
                xs.append(n)
                y = np.median(df_group['Score ' + str(batch_number)])
                print(field, std, y)
                ys.append(y * 100)
                ylows.append(y - np.percentile(df_group['Score ' + str(batch_number)], 2))
                yhighs.append(np.percentile(df_group['Score ' + str(batch_number)], 98) - y)

        label = "$\sigma$=" + str(std)
        ax[p].errorbar(xs, ys, yerr=[ylows, yhighs], label=label, capsize=4, linewidth=1, capthick=1)
        ax[p].set_ylabel('Purity Score')
        ax[p].set_ylim([60, 105])
        ax[p].set_xlabel('Total number of samples')
        ax[p].set_xticks(range(0, 16, 3))
        ax[p].set_yticks(range(60, 101, 5))
        ax[p].legend(loc='lower right')
        # ax[p].set_title(phase_fieldsf[p] + " Phase Field")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('../simulation/graphs/b=1_' + field + '.pdf', dpi=save_dpi)
plt.show()

#table for score vs number of sample:
import pandas as pd
import numpy as np

# Define the std and batch sizes to consider
selected_stds = [0.1, 0.02]
selected_batch_sizes = [1]

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Phase Field', 'Std', 'Batch Size', 'Failure Rate', 'Lb', 'Median', 'Ub'])

# Iterate over phase fields
for field in phase_fields:
    df = pd.read_csv('../simulation/data/' + field + '.csv')
    
    # Calculate failure rate
    df['Failure'] = ~df['End type'].isin(['Terminated success', 'Found'])
    failure_rate = df['Failure'].mean()

    # Filter the DataFrame to include only selected std and batch sizes
    df = df[df['Experimental std'].isin(selected_stds) & df['Batch size'].isin(selected_batch_sizes)]
    
    # Set the multi-index using 'Experimental std' and 'Batch size'
    df.set_index(['Experimental std', 'Batch size'], inplace=True)
    
    # Group by the multi-index and iterate over the groups
    for (std, batch_size), df_group in df.groupby(level=['Experimental std', 'Batch size']):
        scores = df_group.filter(like='Score')
        lb = np.percentile(scores.values.flatten(), 2)
        median = np.median(scores.values.flatten())
        ub = np.percentile(scores.values.flatten(), 98)
        
        # Append results to the DataFrame
        results = results.append({
            'Phase Field': field,
            'Std': std,
            'Batch Size': batch_size,
            'Failure Rate': failure_rate,
            'Lb': lb,
            'Median': median,
            'Ub': ub
        }, ignore_index=True)

# Display the table
print(results)

# Convert to .tbl format for Overleaf
tbl_format = results.to_latex(index=False, header=True, float_format="%.2f")

# Save to file
with open('../simulation/tables/results_table.tbl', 'w') as f:
    f.write(tbl_format)


# Define the std values to consider
selected_stds = [0.1, 0.02]
selected_batch_size = 1

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Phase Field', 'σ', 'Failure Rate', 'Expected Samples to 0.95'])

# Iterate over phase fields
for field in phase_fields:
    df = pd.read_csv('../simulation/data/' + field + '.csv')
    
    # Calculate failure rate
    failure_rate = (~df['End type'].isin(['Terminated success', 'Found'])).mean()

    # Exclude failures
    df = df[df['End type'].isin(['Terminated success', 'Found'])]

    # Filter the DataFrame to include only selected std values and batch size
    df = df[(df['Experimental std'].isin(selected_stds)) & (df['Batch size'] == selected_batch_size)]
    
    # Set the multi-index using 'Experimental std'
    df.set_index(['Experimental std'], inplace=True)
    
    # Group by the multi-index and iterate over the groups
    for std, df_group in df.groupby(level=['Experimental std']):
        first_reach_95 = []

        # Initialize a boolean array to track which rows have been counted
        counted = np.zeros(len(df_group), dtype=bool)

        # Check each column (each batch) for scores reaching 0.95
        for batch_number in range(1, 16):
            batch_col = f'Score {batch_number}'
            if batch_col in df_group.columns:
                batch_reach_95 = df_group[batch_col] >= 0.95
                first_reach_95.extend([batch_number] * np.sum(batch_reach_95 & ~counted))
                # Mark these rows as counted
                counted = counted | batch_reach_95

        # Calculate the expected number of samples if there were any that reached 0.95
        if first_reach_95:
            expected_samples = np.mean(first_reach_95)
        else:
            expected_samples = np.nan  # If no batches reached 0.95, set to NaN

        # Create a DataFrame for the new row
        new_row = pd.DataFrame({
            'Phase Field': [field],
            'σ': [std],
            'Failure Rate': [failure_rate],
            'Expected Samples to 0.95': [expected_samples]
        })
        
        # Append the new row to the results DataFrame using pd.concat
        results = pd.concat([results, new_row], ignore_index=True)

# Display the table
print(results)

# Convert to .tbl format for Overleaf
tbl_format = results.to_latex(index=False, header=True, float_format="%.2f")

# Save to file
with open('results_table.tbl', 'w') as f:
    f.write(tbl_format)

# Plotting the distribution of expected number of samples to reach 0.95 score
plt.figure(figsize=(10, 6))
for label, df_subset in results.groupby('Phase Field'):
    df_subset = df_subset.dropna(subset=['Expected Samples to 0.95'])
    plt.hist(df_subset['Expected Samples to 0.95'], bins=np.arange(1, 17) - 0.5, alpha=0.5, label=label)
plt.xlabel('Expected Number of Samples to Reach Score 0.95')
plt.ylabel('Frequency')
plt.legend(title='Phase Field')
plt.title('Distribution of Expected Number of Samples to Reach Score 0.95')
plt.show()
'''


#batch effect
'''

phase_fields=["MgBOF","MgAlCu","LiAlBO"]
phase_fieldsf=["Mg^{2+}- B^{3+}- O^{2-}- F^-$","Mg-Al-Cu","Li$^{1+}$- Al$^{3+}$- B$^{3+}$- O$^{2-}$"]
#plt.rcParams['figure.figsize'] = (6.50127),2.5
plt.rcParams.update({'font.size': 8})
#cn=-1
#c=['darkred','lightcoral','darkgreen','lightgreen','darkblue','cornflowerblue']
width=6.49
height=3
dpi=600
fig,ax=plt.subplots(1,2,figsize=(width,height),dpi=dpi)
for p,field in enumerate([phase_fields[0]]):
    #cn+=1
    df=pd.read_csv('../simulation/reduced/expanded/'+field+'.csv')
    df=df[(df['End type']=='Terminated success')|(df['End type']=='Found')]
    print(df['End type'].unique())
    print(df.columns)
    for m,std in enumerate([0.1,0.02]):
        for batch_size in [1,3,5]:
            df_red=df[(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            xs=[]
            ys=[]
            ylows=[]
            yhighs=[]
            for batch_number in range(1,16):
                n=batch_size*batch_number
                if n<=15:
                    xs.append(n)
                    y=np.median(df_red['Score '+str(batch_number)])
                    ys.append(y*100)
                    ylows.append(
                        y-np.percentile(df_red['Score '+str(batch_number)],2))
                    yhighs.append(
                        np.percentile(df_red['Score '+str(batch_number)],98)-y)
            label="Batch size = "+str(batch_size)
            ax[m].errorbar(
                xs,ys,yerr=[ylows,yhighs],label=label,capsize=4,linewidth=1,capthick=1)
            ax[m].set_ylabel('Purity Score')uolvpn.liverpool.ac.uk/unmanaged
            ax[m].set_ylim([75,101])
            ax[m].set_xlabel('Total number of samples')
            ax[m].set_xticks(range(0,16,3))
            ax[m].set_yticks(range(75,101,5))
            ax[m].legend(loc='lower right')
            #ax[p].set_title(phase_fieldsf[p]+" Phase Field")
#plt.tight_layout(pad=0)
plt.tight_layout()
plt.savefig('../../paper/batcheffect'+field+'.pdf')
plt.show()
'''

#failure rate table other
'''
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
phase_fields_names=["Mg$^{2+}$-B$^{3+}$-O$^{2-}$-F$^{1-}$","Mg-Al-Cu",
              "Li$^{1+}$-Al$^{3+}$-B$^{3+}$-O$^{2-}$"]
rates=[]
maxs=[]
means=[]
data=[]
for field,name in zip(phase_fields,phase_fields_names):
    df=pd.read_csv('../simulation/reduced/'+field+'.csv')
    print(len(df['End type']=='Nulled late'))
    print(field)
    #for sig in df['Experimental std'].unique():
    for sig in [0.02,0.1]:
        print('sigma ',sig)
        dfa=df[df['Experimental std']==sig]
        #olen=len(dfa)
        #dfa=dfa[dfa['End type']!='Nulled']
        #flen=len(dfa)
        #print('Failure rate is ',(olen-flen)/olen)
        for b in [3]:
        #for b in df['Batch size'].unique():
            print('size ',b)
            dfb=dfa[dfa['Batch size']==b]
            olen=len(dfb)
            dfb=dfb[(dfb['End type']!='Nulled')&
                    (dfb['End type']!='Prior was null')]
            flen=len(dfb)
            rate=(olen-flen)/olen
            print(olen)
            print('Failure rate is ', rate)
            score=dfb['Score 3'].median()
            lb=np.percentile(dfb['Score 3'],2)
            ub=np.percentile(dfb['Score 3'],98)
            data.append([name,sig,round(rate*100),round(lb*100),
                         round(score*100),round(ub*100)])
print('a')
df=pd.DataFrame(data=data,columns=['Phase field','$\sigma$',
                                   'Failure rate',
                                   'LB /\%','M /\%','UB /\%'])
df.style.format({
    'Lower bound': '{:,.0%}\%'.format,
    'Upper bound': '{:,.0%}\%'.format,
    'Failure rate': '{:,.0%}\%'.format,
    'Purity score': '{:,.0%}\%'.format})

print('a')
print(df.head)
#df.to_csv('../figures/failure rate full.csv',index=False)
with open("../../paper/error rate sigma.tbl", "w") as f:

        format = "l" + \
            "@{\hskip 2pt}" +\
            4*"[table-format = 2.2]"

        f.write(df.to_latex(index=False,escape=False))
        '''


#failure rate variation with sigma
'''
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
rates=[]
maxs=[]
means=[]
for field in phase_fields:
    if field==phase_fields[-1]:
        phases=PhaseField(['Li','Al','B','O'])
        ts = phases.triangles
    if field==phase_fields[0]:
        ts=[['Mg 20 B 12 O 36 F 4', 'Mg 6 B 2 O 6 F 6', 'Mg 1 O 1'],
            ['Mg 6 B 2 O 6 F 6', 'Mg 1 O 1', 'Mg 2 F 4'],
            ['Mg 8 B 32 O 56', 'Mg 2 F 4', 'B 6 O 9'],
            ['B 8 F 24', 'Mg 2 F 4', 'B 6 O 9'],
            ['Mg 6 B 2 O 6 F 6', 'Mg 8 B 32 O 56', 'Mg 2 F 4'],
            ['Mg 20 B 12 O 36 F 4', 'Mg 6 B 2 O 6 F 6', 'Mg 8 B 8 O 20'],
            ['Mg 20 B 12 O 36 F 4', 'Mg 6 B 4 O 12', 'Mg 1 O 1'],
            ['Mg 20 B 12 O 36 F 4', 'Mg 6 B 4 O 12', 'Mg 8 B 8 O 20'],
            ['Mg 6 B 2 O 6 F 6', 'Mg 8 B 32 O 56', 'Mg 8 B 8 O 20']]
    if field==phase_fields[1]:
        ts=[['Mg 6 Al 4 Cu 8', 'Mg 2 Al 4 Cu 2', 'Mg 2'],
            ['Mg 8 Cu 4', 'Mg 6 Al 4 Cu 8', 'Mg 2'],
            ['Mg 2 Cu 4', 'Mg 8 Cu 4', 'Mg 6 Al 4 Cu 8'],
            ['Mg 2 Cu 4', 'Mg 6 Al 4 Cu 8', 'Al 2 Cu 6'],
            ['Mg 6 Al 15 Cu 18', 'Mg 6 Al 4 Cu 8', 'Mg 2 Al 4 Cu 2'],
            ['Mg 6 Al 15 Cu 18', 'Mg 6 Al 4 Cu 8', 'Al 16 Cu 36'],
            ['Al 2 Cu 1', 'Mg 2 Al 4 Cu 2', 'Al 1'],
            ['Mg 17 Al 12', 'Mg 2 Al 4 Cu 2', 'Mg 2'],
            ['Mg 17 Al 12', 'Mg 4 Al 8', 'Mg 2 Al 4 Cu 2'],
            ['Mg 6 Al 15 Cu 18', 'Al 2 Cu 1', 'Mg 2 Al 4 Cu 2'],
            ['Mg 6 Al 15 Cu 18', 'Al 2 Cu 1', 'Al 5 Cu 5'],
            ['Mg 4 Al 8', 'Mg 2 Al 4 Cu 2', 'Al 1'],
            ['Mg 6 Al 15 Cu 18', 'Al 16 Cu 36', 'Al 5 Cu 5'],
            ['Mg 6 Al 4 Cu 8', 'Al 16 Cu 36', 'Al 2 Cu 6']]
    calc=wt_converter()
    #print(field)
    df=pd.read_csv('../simulation/2/'+field+'.csv')
    #print("Num Unknowns: ",len(df['Unknown'].unique()))
    df=df.iloc[:3000*27*len(df['Unknown'].unique())]
    for u in df['Unknown'].unique():
        diffs=[]
        for t in ts:
            if u in t:
                #print(u)
                masses=[]
                for p in t:
                    if p!=u:
                        masses.append(calc.get_molar_mass(p)[0])
                avg=(masses[0]+masses[1])/2
                diffs.append(abs(masses[0]-masses[1])/avg)
        #print(diffs)
        dfa=df[df['Unknown']==u]
    #for std in df['Experimental std'].unique():
        #print("STD: ",std)
        #dfa=df[df['Experimental std']==std]
        olen=len(dfa)
        #print("Original length: ",olen)
        dfa=dfa[(dfa['End type']!='Prior was null')]
        olen=len(dfa)
        #print("Original length: ",olen)
        dfa=dfa[(dfa['End type']!='Nulled')]
        #print("Final length: ",len(dfa))
        rate=(olen-len(dfa))/olen
        print("Max: ",np.max(diffs),", Mean: ",np.mean(diffs),', rate: ',
              rate,", Field: ",field)
        rates.append(rate)
        maxs.append(np.max(diffs))
        means.append(np.mean(diffs))
plt.scatter(rates,means)
plt.scatter(rates,maxs)
plt.show()
'''

#plotting expecte num batches
'''
    df=df_a[df_a['Batch size']==s]
    df_found=df[df['End type']=='Found']
    df_found['Score']=0
    for n,i in enumerate(df_found.columns[6:99][::-3]):
        df_found.loc[df_found[i]==0,'Score']=30-n
    print(df_found['Score'].unique())
    print(df_found[df_found['kappa']==100]['Score'].mean())
    print('Mean',df_found[df_found['kappa']==1000]['Score'].mean())
    g=sns.FacetGrid(df_found,col='kappa')
    g.map(sns.violinplot,'Score')
    plt.show()
    '''
#print('a')
#df=pd.read_csv('../simulation/2/LiAlBO.csv')
#print('a')
#df_g=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
#df_g=df[df['End type']=='Terminated successfully']

'''
for m,kappa in enumerate(df_g['Experimental std'].unique()):
    df_red=df_g[
        (df_g['Experimental std']==kappa)
        &(df_g['Batch size']==3)
        &(df_g['Prior size']==3)]
    print(kappa)
    print(len(df_red[df_red['End type']=='Found'])/len(df_red[df_red['End type']=='Terminated successfully']))
    '''
    
'''
                #get first column and last column
                found=False
                for n,i in enumerate(df_red.columns):
                    if label in i:
                        if not found:
                            found=True
                            first_index=n
                        last_index=n
                columns=df_red.columns[first_index:last_index+1][::4]
                for n,i in enumerate(columns):
                    print(n)
                    df=df_red[df_red['End type']=='Found']
                    print(n)
                    sns.distplot(df[i],label='Found')
                    print(n)
                    df=df_red[df_red['End type']=='Terminated successfully']
                    print(n)
                    sns.distplot(df[i],label='Terminated successfully')
                    print(n)
                    plt.legend()
                    print(n)
                    plt.show()
                    print(n)
                    '''
'''
print(df.columns)
for i in df['Experimental std'].unique():
    print(len(df[df['Experimental std']==i]))
print('-')
#define score metric (0=d(nu,u), 1=true second moment, 2=d(closest,u)
df_g=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
df_g=df[df['End type']=='Terminated successfully']
#df_g=df
for score,label in zip(range(4),
                       ['d(mu,u)','true second moment','d(closest u)',
                        'purity']):
    fig,axs=plt.subplots(1,3)
    fig.suptitle(f'Score type is {label}')

    for m,kappa in enumerate(df_g['Experimental std'].unique()):
        ax=axs[m]
        ax.title.set_text(f'Kappa = {kappa}')
        for batch_size in df_g['Batch size'].unique():
            for prior_size in df_g['Prior size'].unique():
                #select sub df
                df_red=df_g[
                    (df_g['Experimental std']==kappa)
                    &(df_g['Batch size']==batch_size)
                    &(df_g['Prior size']==prior_size)]
                
                #get first column and last column
                found=False
                for n,i in enumerate(df_red.columns):
                    if label in i:
                        if not found:
                            found=True
                            first_index=n
                        last_index=n

                print(first_index)
                print(df_red.columns[first_index])
                print(last_index)
                print(df_red.columns[last_index])
                #get score 
                columns=df_red.columns[first_index:last_index+1][::4]
                scores=[]
                scores_std=[]
                for n,i in enumerate(columns):
                    print('xxxx')
                    print(i)
                    result=df_red[i].to_numpy()
                    if score ==3:
                        result*=100
                    scores.append(np.mean(result))
                    scores_std.append(np.std(result))
                print(len(columns))
                print(len(scores))
                print(len(scores_std))
                label=f'Batch size={batch_size},Prior size={prior_size}'
                ax.errorbar(
                    range(len(columns)),scores,yerr=scores_std,label=label,
                    capsize=5)
    plt.legend()
    plt.show()
    '''
#df=pd.read_csv('../simulation/hyperparameters/LiAlBO.csv')
'''
print(df.columns)
df=pd.read_csv('../simulation/2/LiAlBO.csv')
df_g=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
print(len(df_g))
score=3
label='purity'
fig,axs=plt.subplots(1,3)
fig.suptitle(f'Score type is {label}')

for m,kappa in enumerate([50,100,200]):
    ax=axs[m]
    ax.title.set_text(f'demi_len = {kappa}')
    for batch_size in [50,100,200]:
        for prior_size in [50,100,200]:
            #select sub df
            df_red=df_g[
                (df_g['demi_len']==kappa)
                &(df_g['cube_size']==batch_size)
                &(df_g['line_len']==prior_size)]
            print(len(df_red),'7777777777')
            print(kappa,batch_size,prior_size)
            
            #get first column and last column
            found=False
            for n,i in enumerate(df_red.columns):
                if label in i:
                    if not found:
                        found=True
                        first_index=n
                    last_index=n

            print(first_index)
            print(df_red.columns[first_index])
            print(last_index)
            print(df_red.columns[last_index])
            #get score 
            columns=df_red.columns[first_index:last_index+1][::4]
            scores=[]
            scores_std=[]
            for n,i in enumerate(columns):
                result=df_red[i].to_numpy()
                if score ==3:
                    result*=100
                scores.append(np.mean(result))
                scores_std.append(np.std(result))
                print(len(result))
            print(len(columns))
            print(len(scores))
            print(len(scores_std))
            label=f'cube_size={batch_size},line_len={prior_size}'
            ax.errorbar(
                range(len(columns)),scores,yerr=scores_std,label=label,
                capsize=5)

plt.legend()
plt.show()
'''
#somethiing tbc
'''
batch=5
df=pd.read_csv('../simulation/2/LiAlBO.csv')
df=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
label='purity'
print(df['Experimental std'].unique())
for i in df.columns:
    print(i)
print(df['Batch size'].unique())
results=[]
results_columns=['Prior size','Batch size','Measurement std.','Mean purity%',
                 '95 percentile purity']
score=3
g=1
fig,ax=plt.subplots(1)
for m,std in enumerate([0.1,0.02]):
    for prior_size,batch_size in zip([3,3,5,5],[1,5,1,3]):
    #for prior_size in df['Prior size'].unique():
        #for batch_size in [1,5]:
        #for batch_size in df['Batch size'].unique():
            df_red=df[(df['Prior size']==prior_size)
                      &(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            found=False
            for n,i in enumerate(df_red.columns):
                if label in i:
                    if not found:
                        found=True
                        first_index=n
                    last_index=n
            columns=df_red.columns[first_index:last_index+1][::4]
            scores=[]
            scores_std=[]
            scores_low=[]
            scores_high=[]
            for n,i in enumerate(columns[:5]):
                result=df_red[i].to_numpy()
                if score ==3:
                    result*=100
                x=np.mean(result)
                scores.append(x)
                scores_std.append(np.std(result))
                scores_low.append(x-np.percentile(result,g))
                scores_high.append(np.percentile(result,100-g)-x)
            label=(f'Prior size={prior_size},Batch size={batch_size}')
            print('asdf')
            for a,b in zip(scores_low,scores_high):
                print(a,b)
            ax.errorbar(
                range(1,6),scores,yerr=[scores_low,scores_high],
                label=label,capsize=5)

plt.xlabel('Number of batches')
plt.xticks(range(1,6))
plt.ylabel('Purity of new crystal structure')
plt.legend()
plt.show()
'''
#compare to random
'''
batch=5
df=pd.read_csv('../simulation/2/MgBOF.csv')
print(len(df))
df=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
#df=df[(df['End type']!='Prior was null')]
label='purity'
print(df['Experimental std'].unique())
for i in df.columns:
    print(i)
print(df['Batch size'].unique())
results=[]
results_columns=['Prior size','Batch size','Measurement std.','Mean purity%',
                 '95 percentile purity']
score=3
g=2
fig,ax=plt.subplots(1)
for m,std in enumerate(df['Experimental std'].unique()):
    for prior_size in [3]:
        for batch_size in [3]:
            df_red=df[(df['Prior size']==prior_size)
                      &(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            found=False
            for n,i in enumerate(df_red.columns):
                if label in i:
                    if not found:
                        found=True
                        first_index=n
                    last_index=n
            columns=df_red.columns[first_index:last_index+1][::4]
            scores=[]
            scores_std=[]
            scores_low=[]
            scores_high=[]
            for n,i in enumerate(columns[:5]):
                result=df_red[i].to_numpy()
                if score ==3:
                    result*=100
                x=np.mean(result)
                scores.append(x)
                scores_std.append(np.std(result))
                scores_low.append(x-np.percentile(result,g))
                scores_high.append(np.percentile(result,100-g)-x)
            label=(f'Prior size={prior_size},Batch size={batch_size}'
                   +f', Measurement $\sigma$={std}')
            print('asdf')
            for a,b in zip(scores_low,scores_high):
                print(a,b)
            ax.errorbar(
                range(1,6),scores,yerr=[scores_low,scores_high],
                label=label,capsize=5)
df=pd.read_csv('../simulation/testing/MgBOF.csv')
print(len(df))
#df=df[(df['End type']!='Prior was null')]
df=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
score=3
g=2
for m,std in enumerate(df['Experimental std'].unique()):
    for prior_size in [3]:
        for batch_size in [3]:
            df_red=df[(df['Prior size']==prior_size)
                      &(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            found=False
            for n,i in enumerate(df_red.columns):
                if label in i:
                    if not found:
                        found=True
                        first_index=n
                    last_index=n
            columns=df_red.columns[first_index:last_index+1][::4]
            scores=[]
            scores_std=[]
            scores_low=[]
            scores_high=[]
            for n,i in enumerate(columns[:5]):
                result=df_red[i].to_numpy()
                if score ==3:
                    result*=100
                x=np.mean(result)
                scores.append(x)
                scores_std.append(np.std(result))
                scores_low.append(x-np.percentile(result,g))
                scores_high.append(np.percentile(result,100-g)-x)
            label=(f'RANDOM,Prior size={prior_size},Batch size={batch_size}'
                   +f', Measurement $\sigma$={std}')
            for a,b in zip(scores_low,scores_high):
                print(a,b)
            ax.errorbar(
                range(1,6),scores,yerr=[scores_low,scores_high],
                label=label,capsize=5)

plt.xlabel('Number of batches')
plt.xticks(range(1,6))
plt.ylabel('Purity of new crystal structure')
plt.legend()
plt.show()
'''
#pe scan
'''
batch=5
df=pd.read_csv('../simulation/testing/MgBOF_p_e.csv')
print(len(df))
df=df[(df['End type']!='Nulled')&(df['End type']!='Prior was null')]
#df=df[(df['End type']!='Prior was null')]
label='purity'
print(df['Experimental std'].unique())
for i in df.columns:
    print(i)
print(df['Batch size'].unique())
results=[]
results_columns=['Prior size','Batch size','Measurement std.','Mean purity%',
                 '95 percentile purity']
score=3
g=2
fig,ax=plt.subplots(1)
for m,std in enumerate(df['Experimental std'].unique()):
    for prior_size in df['Initial predicted error'].unique():
        for batch_size in [3]:
            df_red=df[(df['Initial predicted error']==prior_size)
                      &(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            found=False
            for n,i in enumerate(df_red.columns):
                if label in i:
                    if not found:
                        found=True
                        first_index=n
                    last_index=n
            columns=df_red.columns[first_index:last_index+1][::4]
            scores=[]
            scores_std=[]
            scores_low=[]
            scores_high=[]
            for n,i in enumerate(columns[:5]):
                result=df_red[i].to_numpy()
                if score ==3:
                    result*=100
                x=np.mean(result)
                scores.append(x)
                scores_std.append(np.std(result))
                scores_low.append(x-np.percentile(result,g))
                scores_high.append(np.percentile(result,100-g)-x)
            label=(f'Prior size={prior_size},Batch size={batch_size}'
                   +f', Measurement $\sigma$={std}')
            print('asdf')
            for a,b in zip(scores_low,scores_high):
                print(a,b)
            ax.errorbar(
                range(1,6),scores,yerr=[scores_low,scores_high],
                label=label,capsize=5)
plt.xlabel('Number of batches')
plt.xticks(range(1,6))
plt.ylabel('Purity of new crystal structure')
plt.legend()
plt.show()
'''
#code for plotting score at n'th batch
phase_fields=["MgBOF","MgAlCu","LiAlBO"]
phase_fieldsf=["Mg^{2+}- B^{3+}- O^{2-}- F^-$","Mg-Al-Cu","Li$^{1+}$- Al$^{3+}$- B$^{3+}$- O$^{2-}$"]
#plt.rcParams['figure.figsize'] = (6.50127),2.5
plt.rcParams.update({'font.size': 8})
#cn=-1
#c=['darkred','lightcoral','darkgreen','lightgreen','darkblue','cornflowerblue']
width=6.49
height=3
dpi=600
fig,ax=plt.subplots(1,3,figsize=(width,height),dpi=dpi)
for p,field in enumerate(phase_fields):
    #cn+=1
    df=pd.read_csv('../simulation/reduced/expanded/'+field+'.csv')
    print(df.head())
    '''

    df=df[(df['End type']=='Terminated success')|(df['End type']=='Found')]
    print(df['End type'].unique())
    print(df.columns)
    for m,std in enumerate([0.1,0.02]):
        for batch_size in [1]:#,3,5]:
            df_red=df[(df['Experimental std']==std)
                      &(df['Batch size']==batch_size)]
            xs=[]
            ys=[]
            ylows=[]
            yhighs=[]
            for batch_number in range(1,16):
                n=batch_size*batch_number
                if n<=15:
                    xs.append(n)
                    y=np.median(df_red['Score '+str(batch_number)])
                    print(field,std,y)
                    ys.append(y*100)
                    ylows.append(
                        y-np.percentile(df_red['Score '+str(batch_number)],2))
                    yhighs.append(
                        np.percentile(df_red['Score '+str(batch_number)],98)-y)
            label="$\sigma$="+str(std)
            ax[p].errorbar(
                xs,ys,yerr=[ylows,yhighs],label=label,capsize=4,linewidth=1,capthick=1)
            ax[p].set_ylabel('Purity Score')
            ax[p].set_ylim([60,105])
            ax[p].set_xlabel('Total number of samples')
            ax[p].set_xticks(range(0,16,3))
            ax[p].set_yticks(range(60,101,5))
            ax[p].legend(loc='lower right')
            #ax[p].set_title(phase_fieldsf[p]+" Phase Field")
#plt.tight_layout(pad=0)
plt.tight_layout()
plt.savefig('../../paper/b=1_'+field+'.pdf')
plt.show()
'''



