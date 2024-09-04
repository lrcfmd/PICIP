from experimental_phase_fields import *
from visualise_square import *
from algorithm import *
from experimental_phase_fields import *

#code for mpds data
#setup
'''
field=PhaseField(['Li','Al','B','O'])
model=all_information()
basis=np.array([[-0.66714371, -0.24907388,  0.64936011,  0.26685749],
       [ 0.5488059 , -0.71072244,  0.38143889, -0.21952236]])
species_dict={'Li':1,'Al':3,'B':3,'O':-2}
model.setup_from_species_dict(species_dict,basis=basis)
plot=Square(model)
plot.make_binaries_as_corners()
plot.create_fig(corner_text=False)
#plot lines
for ts in field.triangles:
    for n,ta in enumerate(ts):
        for tb in ts[n+1:]:
            a=field.pure_phases[ta]
            b=field.pure_phases[tb]
            plot.plot_line(a,b,convert=True)
#label
xs=[]
ys=[]
labels=[]
data=[]
for k,v in field.pure_phases.items():
    label=k.replace(" ","")
    p=model.convert_point_to_constrained(v)
    x=p[0]
    y=p[1]
    data.append([x,y,label])
df=pd.DataFrame(columns=['x','y','label'],data=data)
plot.custom_scatter(df)

plot.show(show=False,saveimg="../figures/test",save="../figures/test")
print('Done')
sys.exit()
'''
#code for pymatgen data
#setup
'''
model=all_information()
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
cube_size=50
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()
print(model.contained_point,'check norm')
print(model.convert_point_to_constrained(model.contained_point))
c=model.contained_point
x=[1,0,0,2]
x=np.array(x)/sum(x)
b=model.basis
x_s=b@x
c_s=b@c
x_sc=x_s-c_s
xc=model.convert_point_to_constrained(x)
print(f'x={x}, x_c={xc}, x_sc={x_sc}')
print(model.cube_size)
a=model.convert_to_standard_basis([0,0])-model.convert_to_standard_basis([0,1.037])
b=model.convert_to_standard_basis([0,0])-model.convert_to_standard_basis([1,0])
print(a/np.linalg.norm(a))
print(b/np.linalg.norm(b))
print(sum(a))


labels=[]
data=[]
indexes=[]
v=[]
triangles=[]
cube_size=400
for x in model.omega:
    comp=model.simulator.get_comp(x)
    ps=[]
    ws=[]
    for k,v in model.simulator.pd.get_decomposition(Composition(comp)).items():
        if v>0.001:
            ps.append(k)
    #ps=[x if y>0.0001 else None for x,y in zip(ps,ws)]
    if len(ps)==3:
        triangle=[x.composition for x in ps]
        if not triangle in triangles:
            triangles.append(triangle)
    for k in ps:
        label=k.composition
        rep=[k.composition.get_atomic_fraction(Element(x)) for x in
             species_dict]
        rep_con=model.convert_point_to_constrained(rep)
        if label not in labels:
            labels.append(label)
            print(label.reduced_formula)
            x=rep_con[0]
            y=rep_con[1]
            data.append([cube_size*x/15,cube_size*y/15,label.reduced_formula])
#for n,t in enumerate(model.simulator.triangles):
#    print(n,t)
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()

plot=Square(model)
plot.make_binaries_as_corners()
plot.create_fig(corner_text=False,lw=2)
plot.fig.update_traces(marker_size=0.1,marker_color='black')

#explanation figure
#plot lines
for ts in triangles:
    for n,ta in enumerate(ts):
        a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
        for tb in ts[n+1:]:
            plot_line=True
            b=[tb.get_atomic_fraction(Element(x)) for x in species_dict]
            if str(tb)=='Mg2 F4' and str(ta)=='Mg1 O1':
                    plot_line=True
            if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg2 F4':
                    plot_line=True
            if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg1 O1':
                    plot_line=True
            if str(tb)=='Mg6 B2 O6 F6' or str(ta)=='Mg6 B2 O6 F6':
                    plot_line=False

            if plot_line:
                pass
                #plot.plot_line(a,b,w=1,convert=True)
df=pd.DataFrame(columns=['x','y','label'],data=data)

ps=model.simulator.pd.get_decomposition(Composition('Mg15B15O5F65'))
#triangle=[x.composition for x in ps]
#a=[p.get_atomic_fraction(Element(x)) for x i/home/danny/bitmap.pngn species_dict]

plot.fig.update_traces(line_width=10)
lw=12
#lw=48

a=[6,2,6,6]
a_con=model.convert_point_to_constrained(a)

#b=[20,12,36,4]
#b_con=model.convert_point_to_constrained(b)
#plotter.plot_point(
#    b_con,label='Mg<sub>5</sub>(BO<sub>3</sub>)<sub>3</sub>F',
#    c='blue',s=10)

b=[1,0,0,2]
b_con=model.convert_point_to_constrained(b)

c=[1,0,1,0]
c_con=model.convert_point_to_constrained(c)

s_con=0.2*a_con+0.3*b_con+0.5*c_con
#plot.plot_point(
#    s_con,label='Sampled composition',
#    s=100,symbol='133')
print(plot.get_composition_string([s_con]))
print(s_con)
print(model.convert_to_standard_basis(s_con))


w=np.array([0.3,0.5])/0.8
print('w',w)
k_con=w[0]*b_con+w[1]*c_con
k_con=k_con
#plot.plot_point(
#    k_con,label='Average known',s=100,symbol='133')
print(plot.get_composition_string([k_con]))

print(f'kcon: {k_con}, kstan: {model.convert_to_standard_basis(k_con)}')
print(f'scon: {s_con}, sstan: {model.convert_to_standard_basis(s_con)}')
print('sub con',model.convert_to_standard_basis((s_con-k_con)/np.linalg.norm(s_con-k_con)))
ed_stan=model.convert_to_standard_basis(s_con)-model.convert_to_standard_basis(k_con)
ed_stan/=np.linalg.norm(ed_stan)
print(np.dot(ed_stan,[2,3,-2,-1]))
print(np.linalg.norm(ed_stan))
print(ed_stan)
wtconvert=wt_converter()
a=0.375*(24.305+38)
b=0.625*(24.305+16)
w=np.array([a,b])/sum([a,b])
moles, moles_error, formulas_standard=wtconvert.wt_to_moles(
    ['Mg F 2','Mg O'],w,phase_field=species_dict.keys(),weights_error=[0.02])
print(f"{moles=} {moles_error=}")
mean,sig=model.compute_k_sig(moles,formulas_standard,moles_error)
print(f'mean: {mean}, sig: {sig}, cube:{model.cube_size}')
'''
'''

model.add_sksig(s_con,k_con,0.1+np.ones((2,2)))
model.set_end_points_from_s()

e=model.end_points[0]
plot.plot_line(s_con,e,w=18,c='green')#,dash='dot')

plot.custom_scatter(
    df[df['label']=='Mg3B(OF)3'],s=100,c='red',symbol=['134'])
plot.custom_scatter(df[df['label']!='Mg3B(OF)3'],s=100,c='black')

plot.fig.update_xaxes(
    #title_text=("x<sup>'</sup><sub>0</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>0</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=140,
    constrain='domain',
    #showticklabels=False,
    #side='bottom',
)

plot.fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    #title_text=("x'<sub>1</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>1</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=140,
    constrain='domain',
    #showticklabels=False
)


#plot.fig.update_yaxes(range=[0.35,1.08])
#plot.fig.update_xaxes(range=[-0.56,0.01])
plot.fig.update_yaxes(range=[-0.05,1.1])
plot.fig.update_xaxes(range=[-0.6,0.42])
img=('../paper/search',4000,4000)
plot.show(img=img,s=80,show=False)
#plot.fig.update_traces(marker_size=2)
#img=('../paper/plane',4000,4000)
#plotter.show(img=img)


#plot.show(saveimg="../figures/test",save="../figures/test")
print('Done')
sys.exit()
'''
#error
model=all_information()
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
cube_size=50
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()

labels=[]
data=[]
indexes=[]
v=[]
triangles=[]
cube_size=400
for x in model.omega:
    comp=model.simulator.get_comp(x)
    ps=[]
    ws=[]
    for k,v in model.simulator.pd.get_decomposition(Composition(comp)).items():
        if v>0.001:
            ps.append(k)
    #ps=[x if y>0.0001 else None for x,y in zip(ps,ws)]
    if len(ps)==3:
        triangle=[x.composition for x in ps]
        if not triangle in triangles:
            triangles.append(triangle)
    for k in ps:
        label=k.composition
        rep=[k.composition.get_atomic_fraction(Element(x)) for x in
             species_dict]
        rep_con=model.convert_point_to_constrained(rep)
        if label not in labels:
            labels.append(label)
            print(label.reduced_formula)
            x=rep_con[0]
            y=rep_con[1]
            data.append([cube_size*x/15,cube_size*y/15,label.reduced_formula])


model=all_information()
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
cube_size=500
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()
plot=Square(model)
plot.make_binaries_as_corners()
plot.create_fig(corner_text=False,lw=2)
plot.fig.update_traces(marker_size=0.1,marker_color='black')

'''
for ts in triangles:
    for n,ta in enumerate(ts):
        a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
        for tb in ts[n+1:]:
            plot_line=True
            b=[tb.get_atomic_fraction(Element(x)) for x in species_dict]
            if str(tb)=='Mg2 F4' and str(ta)=='Mg1 O1':
                    plot_line=True
            if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg2 F4':
                    plot_line=True
            if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg1 O1':
                    plot_line=True
            if str(tb)=='Mg6 B2 O6 F6' or str(ta)=='Mg6 B2 O6 F6':
                    plot_line=False

            if plot_line:
                pass
                #plot.plot_line(a,b,w=4,convert=True)
                '''
df=pd.DataFrame(columns=['x','y','label'],data=data)

ps=model.simulator.pd.get_decomposition(Composition('Mg15B15O5F65'))
#triangle=[x.composition for x in ps]
#a=[p.get_atomic_fraction(Element(x)) for x i/home/danny/bitmap.pngn species_dict]

#plot.fig.update_traces(line_width=10)
#lw=48
a=[6,2,6,6]
a_con=model.convert_point_to_constrained(a)

#b=[20,12,36,4]
#b_con=model.convert_point_to_constrained(b)
#plotter.plot_point(
#    b_con,label='Mg<sub>5</sub>(BO<sub>3</sub>)<sub>3</sub>F',
#    c='blue',s=10)

b=[1,0,0,2]
b_con=model.convert_point_to_constrained(b)

c=[1,0,1,0]
c_con=model.convert_point_to_constrained(c)

s_con=0.2*a_con+0.3*b_con+0.5*c_con

w=np.array([0.3,0.5])/0.8
a=w[0]-0.02
b=w[1]+0.02
z=[a,b]
k_con=z[0]*b_con+z[1]*c_con
k_stan=model.convert_to_standard_basis(k_con)

w=np.array([0.3,0.5])/0.8
a=0.375*(24.305+38)
b=0.625*(24.305+16)
w=np.array([a,b])/sum([a,b])
w=[w[0]+0.02,w[1]-0.02]
a=w[0]/(24.305+38)
b=w[1]/(24.305+16)
z=np.array([a,b])/sum([a,b])
k_con2=z[0]*b_con+z[1]*c_con
wtconvert=wt_converter()
moles, moles_error, formulas_standard=wtconvert.wt_to_moles(
    ['Mg F 2','Mg O'],w,phase_field=species_dict.keys(),weights_error=[0.01])
mean,sig=model.compute_k_sig(moles,formulas_standard,moles_error)
Sigma=sig
#Sigma=np.eye(2)*(0.005*cube_size)**2
model.demi_len=500
model.line_len=500
model.use_fib=True
model.add_sksig(s_con,mean,Sigma,make_p=True)
model.set_end_points_from_s()

e=model.end_points[0]
#plot.plot_line(s_con,e,w=5,c='green')#,dash='dot')

#plot.custom_scatter(df[df['label']!='Mg3B(OF)3'],s=20,c='black')

Sigma=sig
#Sigma=np.eye(2)*cube_size*0.1
print(f'Sigma ball {Sigma}')
m = 300
X = np.linspace(mean[0]-0.1*cube_size, mean[0]+0.1*cube_size, m)
Y = np.linspace(mean[1]-0.1*cube_size, mean[1]+0.1*cube_size, m)
X, Y = np.meshgrid(X, Y)

mu=mean
n = mu.shape[0]

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

Sigma_det = np.linalg.det(Sigma)
Sigma_inv = np.linalg.inv(Sigma)
N = np.sqrt((2*np.pi)**n * Sigma_det)
# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
# way across all the input variables.
fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

z=np.exp(-fac / 2) / N
posr=np.reshape(pos,(m**2,2))
zr=np.reshape(z,(m**2))
p=scipy.interpolate.griddata(posr,zr,model.omega,method='linear',fill_value=0)
p1=len(p)*p/sum(p)/500
#plot.plot_points({'k':k_stan})
mean_stan=model.convert_to_standard_basis(mean)
#plot.plot_points({'mean':mean_stan})


plot.plot_p(model.omega,p1,s=2)#,cmap='viridis')
#plot.show()
#square.plot_constraints(ys,x,ax=ax,ms=2)


#ax.plot(mean[0],mean[1],marker='x',c='red',label='sample')
#ax.plot(avg_known[0],avg_known[1],marker='o',c='red',label='avg_known')


p=model.values
p=p/sum(p)

a=model.omega-s_con
b=s_con-mean
c=np.dot(a,b)
points=np.delete(model.omega,np.where(np.dot(a,b)<=0),axis=0)
p=np.delete(p,np.where(np.dot(a,b)<=0),axis=0)
p=len(p)*p/sum(p)
print('here')

print(points,'jjjjjj')
plot.plot_p(points,p,s=2)#,s=0.5,cmap='viridis',zorder=5)
#plot.plot_points_np(points,s=2)#,s=0.5,cmap='viridis',zorder=5)
#plt.savefig('../../paper/ball_conea.png',dpi=20)
#plt.savefig('../../paper/ball_coneb.png',dpi=100)
#print('a')
#plt.savefig('../../paper/ball_conec.png',dpi=1000)
y=np.array([-0.07,1.1])*cube_size
x=np.array([-0.63,0.42])*cube_size
plot.fig.update_xaxes(range=x)
plot.fig.update_yaxes(range=y)
lw=2
plot.fig.update_xaxes(
    #title_text=("x<sup>'</sup><sub>0</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>0</sub>"
    #            ),
    showline=False,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=8,
    constrain='domain',
    zeroline=False,
    showticklabels=False,
    visible=False,
    #side='bottom',
)

plot.fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    #title_text=("x'<sub>1</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>1</sub>"
    #            ),
    showline=False,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=8,
    constrain='domain',
    zeroline=False,
    showticklabels=False,
    visible=False
)


#plot.fig.update_yaxes(range=[0.35,1.08])
#plot.fig.update_xaxes(range=[-0.56,0.01])
#plot.fig.update_yaxes(range=[-0.05,1.1])
#plot.fig.update_xaxes(range=[-0.6,0.42])
img=('../paper/search',4000,4000)
plot.fig.update_traces(showlegend=False)
plot.show(s=14,show=True)
plot.fig.write_image('../simulation/field_diagrams/test.png',scale=3)






'''
plot.fig.update_xaxes(
    #title_text=("x<sup>'</sup><sub>0</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>0</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=t,
    constrain='domain',
    #showgrid=False,
    #showticklabels=False,
    #side='bottom',
)

plot.fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    #title_text=("x'<sub>1</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>1</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=t,
    constrain='domain',
    #showgrid=False,
    #showticklabels=False
)
'''

'''
plot.custom_scatter(
    df[df['label']=='Mg3B(OF)3'],s=30,w=4,c='black',symbol=['134'])
plot.plot_points({'u':[3,1,3,3]})
plot.fig.update_layout(showlegend=True)
#plot.fig.data = plot.fig.data[::-1]
#plot.fig.update_yaxes(range=[0.35,1.08])
#plot.fig.update_xaxes(range=[-0.56,0.01])
img=('../../paper/uonly',4000,4000)

lw=3
uolvpn.liverpool.ac.uk/unmanagedt=35

mm_width = 166/2
dpi = 300
inch_width = mm_width / 25.4
pixel_width = int(inch_width * dpi)
y=np.array([-0.07,1.1])*cube_size
x=np.array([-0.63,0.42])*cube_size
dy=y[1]-y[0]
dx=x[1]-x[0]
height_over_width=dy/dx
pixel_height=pixel_width*height_over_width*1
#plot.fig.update_layout(width=pixel_width,showlegend=False)
plot.fig.update_layout(height=pixel_height,width=pixel_width,showlegend=True)

plot.fig.update_xaxes(range=x,
                      #showline=False,
                      #showticklabels=False,
                      linewidth=lw,
                      linecolor='black',
                      zeroline=False,
                      showgrid=False)
plot.fig.update_yaxes(range=y,
                      #showline=False,
                      #scaleanchor = "x",
                      #scaleratio = 1,
                      #showticklabels=False,
                      linewidth=lw,
                      linecolor='black',
                      zeroline=False,
                      showgrid=False)
#plot.show(img=img,s=t,show=False,transparent=True)
#plot.fig.update_traces(marker_size=2)
#img=('../paper/plane',4000,4000)
plot.show(img=img)


#plot.show(saveimg="../figures/test",save="../figures/test")
sys.exit()
'''

#code for plotting diagram with unknown removed and area - MgBOF
'''
model=all_information()
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
cube_size=50
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
areas=[]
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()

labels=[]
data=[]
indexes=[]
v=[]
triangles=[]
for x in model.omega:
    comp=model.simulator.get_comp(x)
    ps=[]
    ws=[]
    for k,v in model.simulator.pd.get_decomposition(Composition(comp)).items():
        if v>0.001:
            ps.append(k)
    #ps=[x if y>0.0001 else None for x,y in zip(ps,ws)]
    if len(ps)==3:
        triangle=[x.composition for x in ps]
        if not triangle in triangles:
            triangles.append(triangle)
    for k in ps:
        label=k.composition
        rep=[k.composition.get_atomic_fraction(Element(x)) for x in
             species_dict]
        rep_con=model.convert_point_to_constrained(rep)
        if label not in labels:
            labels.append(label)
            print(label.reduced_formula)
            x=rep_con[0]
            y=rep_con[1]
            data.append([x/cube_size,y/cube_size,label.reduced_formula])

cube_size=1
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)
model.add_simulator_c()


us=["Mg6 B2 O6 F6","Mg20 B12 O36 F4"]
for u in us:
    print('-------------------')
    print(u)
    area=0
    plot=Square(model)
    plot.make_binaries_as_corners()
    plot.create_fig(corner_text=False,lw=2)
    plot.fig.update_traces(marker_size=0.1,marker_color='black')
    for ts in triangles:
        contains_u=False
        for n,ta in enumerate(ts):
            a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
            if str(ta)==u:
                contains_u=True
            for tb in ts[n+1:]:
                plot_line=True
                b=[tb.get_atomic_fraction(Element(x)) for x in species_dict]
                if str(ta)==u or str(tb)==u:
                    plot_line=False

                if plot_line:
                    plot.plot_line(a,b,w=1,convert=True)
        if contains_u:
            x=[[v.get_atomic_fraction(Element(x)) for x in species_dict]
               for v in ts]
            a=model.convert_point_to_constrained(x[0])
            b=model.convert_point_to_constrained(x[1])
            c=model.convert_point_to_constrained(x[2])
            ab=b-a
            ac=c-a
            lab1=np.linalg.norm(ab)
            lac1=np.linalg.norm(ac)
            areaA=0.5*np.cross(ab,ac)
            print(f'{ts} contains {u} with area {areaA}')
            area+=abs(areaA)
        else:
            print(f'{ts} did not contain {u}')
    print(area)

    df=pd.DataFrame(columns=['x','y','label'],data=data)
    print(df['label'].unique())
    if u=='Mg6 B2 O6 F6':
        plot.custom_scatter(
            df[df['label']=='Mg3B(OF)3'],s=25,w=4,c='red',symbol=['134'])
        plot.custom_scatter(
            df[df['label']!='Mg3B(OF)3'],s=20,w=4,c='black')
    else:
        plot.custom_scatter(
            df[df['label']=='Mg5B3O9F'],s=25,w=4,c='red',symbol=['134'])
        plot.custom_scatter(
            df[df['label']!='Mg5B3O9F'],s=20,w=4,c='black')

    plot.fig.update_traces(line_width=4)
    axis_lw=3

    img=('../../paper/_'+str(u),4000,4000)
    t=35

    mm_width = 166/2
    dpi = 300
    inch_width = mm_width / 25.4
    pixel_width = int(inch_width * dpi)
    y=np.array([-0.07,1.1])*cube_size
    x=np.array([-0.63,0.42])*cube_size
    dy=y[1]-y[0]
    dx=x[1]-x[0]
    height_over_width=dy/dx
    pixel_height=pixel_width*height_over_width*1
    #plot.fig.update_layout(width=pixel_width,showlegend=False)
    plot.fig.update_layout(height=pixel_height,width=pixel_width,showlegend=True)

    plot.fig.update_xaxes(range=x,
                          #showline=False,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          showgrid=False)
    plot.fig.update_yaxes(range=y,
                          #showline=False,
                          #scaleanchor = "x",
                          #scaleratio = 1,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          showgrid=False)
    #plot.show(img=img,s=t,show=False,transparent=True)
    #plot.fig.update_traces(marker_size=2)
    #img=('../paper/plane',4000,4000)
    plot.show(img=img,s=25)
    areas.append(area)


    #plot.show(saveimg="../figures/test",save="../figures/test")
print(areas)
sys.exit()
'''
#code for plotting phase with removed u: LiAlBO
'''
model=all_information()
species_dict={'Li':1,'Al':3,'B':3,'O':-2}
cube_size=1
basis=np.array([[ 0,0.70710678,-0.70710678,0],
                [0.86386843,-0.25916053,-0.25916053,-0.34554737]])
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)

#find nice basis
#left=[0,0,2,3]
#right=[0,2,0,3]
#left=np.array(left)/sum(left)
#right=np.array(right)/sum(right)
#x=right-left
#x=x/np.linalg.norm(x)
#print(model.normal_vectors)
#A=np.vstack([model.normal_vectors,x])
#print(A)
#y=model.find_orthonormal(A.T)
#print(y)
#basis=np.array([x,y])
#print(basis)
#model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis)

phase_field=list(species_dict.keys())
e_data=PhaseField(phase_field)
model.add_simulator_e(e_data.pure_phases,e_data.triangles)


us=["Li 2 Al B 5 O 10",
    "Li Al B 2 O 5",
    "Li 3 Al B 2 O 6",
    "Li 2 Al B O 4",
    "Li Al 7 B 4 O 17",
    "Li 2.46 Al 0.18 B O 3"]
areas=[]
for u in us:
    plot=Square(model)
    plot.make_binaries_as_corners()
    plot.create_fig(corner_text=False,lw=2)
    area=0
    for tri in model.simulator.triangles:
        contains_u=False
        for n,p1 in enumerate(tri):
            for p2 in tri[n+1:]:
                p1_c=model.simulator.pure_phases[p1]
                p2_c=model.simulator.pure_phases[p2]
                if p1==u or p2==u:
                    contains_u=True
                else:
                    plot.plot_line(p1_c,p2_c,w=0.66)

        if contains_u:
            #get triangle area
            x=[model.simulator.pure_phases[p] for p in tri]
            a=x[0]
            b=x[1]
            c=x[2]
            ab=b-a
            ac=c-a
            lab1=np.linalg.norm(ab)
            lac1=np.linalg.norm(ac)
            areaA=0.5*np.cross(ab,ac)
            area+=abs(areaA)/cube_size**2

    for k,v in model.simulator.pure_phases.items():
        if k==u:
            plot.plot_points_np([v],c='red',s=6,w=1,m='134')
        else:
            plot.plot_points_np([v],c='black',s=5,w=4)
    print(f'Area for {u} is {area:.3f}')
    areas.append(area)

    axis_lw=1


    y=np.array([-0.1,0.9])*cube_size
    x=np.array([-0.65,0.1])*cube_size
    plot.fig.update_xaxes(range=x)
    plot.fig.update_yaxes(range=y)

    plot.fig.update_xaxes(#showline=False,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          constrain='domain',
                          showgrid=False)
    plot.fig.update_yaxes(#showline=False,
                          scaleanchor = "x",
                          scaleratio = 1,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          constrain='domain',
                          showgrid=False)
    #plot.show(img=img,s=t,show=False,transparent=True)
    #plot.fig.update_traces(marker_size=2)
    #img=('../paper/plane',4000,4000)
    plot.show(s=8,title=f'{area=:.3f} for {u}')
    plot.fig.write_image('../simulation/field_diagrams/'+u+".png",scale=3)




    #plot.show(saveimg="../figures/test",save="../figures/test")
print(areas)
sys.exit()
'''

#code for plotting diagram with unknown removed and area - MgAlCU
'''
model=all_information()
species_dict=['Mg','Al','Cu']
cube_size=50
cp=[1/3,1/3,1/3]
basis=np.array([[0.,0.70710678,-0.70710678],
                [0.81649658,-0.40824829 ,-0.40824829]])
areas=[]
model.setup_b(
    species_dict,cube_size=cube_size,basis=basis,contained_point=cp)
#find nice basis
#left=[0,0,1]
#right=[0,1,0]
#left=np.array(left)/sum(left)
#right=np.array(right)/sum(right)
#x=right-left
#x=x/np.linalg.norm(x)
#print(model.normal_vectors)
#A=np.vstack([model.normal_vectors,x])
#print(A)
#y=model.find_orthonormal(A.T)
#print(y)
#basis=np.array([x,y])
#print(basis)

model.add_simulator_c()
us=['Mg6 Al4 Cu8','Mg2 Al4 Cu2','Mg6 Al15 Cu18']
us_print=["Mg3(AlCu2)2","MgAl2Cu","Mg2Al5Cu6"]

labels=[]
data=[]
indexes=[]
v=[]
triangles=[]
for x in model.omega:
    comp=model.simulator.get_comp(x)
    ps=[]
    ws=[]
    for k,v in model.simulator.pd.get_decomposition(Composition(comp)).items():
        if v>0.001:
            ps.append(k)
    #ps=[x if y>0.0001 else None for x,y in zip(ps,ws)]
    if len(ps)==3:
        triangle=[x.composition for x in ps]
        if not triangle in triangles:
            triangles.append(triangle)
    for k in ps:
        label=k.composition
        rep=[k.composition.get_atomic_fraction(Element(x)) for x in
             species_dict]
        rep_con=model.convert_point_to_constrained(rep)
        if label not in labels:
            labels.append(label)
            print(label.reduced_formula)
            x=rep_con[0]
            y=rep_con[1]
            data.append([x/cube_size,y/cube_size,label.reduced_formula])

cube_size=1
model.setup_b(
    species_dict,cube_size=cube_size,basis=basis,contained_point=cp)
model.add_simulator_c()


for u,u_print in zip(us,us_print):
    print('-------------------')
    print(u)
    area=0
    plot=Square(model)
    #plot.make_binaries_as_corners()
    plot.create_fig(corner_text=False,lw=2,corners=False)
    plot.fig.update_traces(marker_size=0.1,marker_color='black')
    for ts in triangles:
        contains_u=False
        for n,ta in enumerate(ts):
            a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
            if str(ta)==u:
                contains_u=True
            for tb in ts[n+1:]:
                plot_line=True
                b=[tb.get_atomic_fraction(Element(x)) for x in species_dict]
                if str(ta)==u or str(tb)==u:
                    plot_line=False

                if plot_line:
                    plot.plot_line(a,b,w=0.66,convert=True)
        if contains_u:
            x=[[v.get_atomic_fraction(Element(x)) for x in species_dict]
               for v in ts]
            a=model.convert_point_to_constrained(x[0])
            b=model.convert_point_to_constrained(x[1])
            c=model.convert_point_to_constrained(x[2])
            ab=b-a
            ac=c-a
            lab1=np.linalg.norm(ab)
            lac1=np.linalg.norm(ac)
            areaA=0.5*np.cross(ab,ac)
            print(f'{ts} contains {u} with area {areaA}')
            area+=abs(areaA)
        else:
            print(f'{ts} did not contain {u}')
    print(area)

    df=pd.DataFrame(columns=['x','y','label'],data=data)
    print(df['label'].unique())
    plot.custom_scatter(
        df[df['label']==u_print],s=6,w=1,c='red',symbol=['134'])
    plot.custom_scatter(
        df[df['label']!=u_print],s=5,w=4,c='black')

    axis_lw=1

    y=np.array([-0.55,0.95])*cube_size
    x=np.array([-0.9,0.9])*cube_size
    plot.fig.update_xaxes(range=x)
    plot.fig.update_yaxes(range=y)


    plot.fig.update_xaxes(#showline=False,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          constrain='domain',
                          showgrid=False)
    plot.fig.update_yaxes(#showline=False,
                          scaleanchor = "x",
                          scaleratio = 1,
                          #showticklabels=False,
                          linewidth=axis_lw,
                          linecolor='black',
                          zeroline=False,
                          constrain='domain',
                          showgrid=False)
    plot.show(s=8,title=f'{area=:.3f} for {u}')
    areas.append(area)
    plot.fig.write_image('../simulation/field_diagrams/'+u+".png",scale=3)
print(areas)
sys.exit()
'''
#code for demonstrating affect of line between knowns length
'''
model=all_information()
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
cube_size=50
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
contained_point=[0,2,3,0]
model.setup_from_species_dict(species_dict,cube_size=cube_size,basis=basis,
                             contained_point=contained_point)
model.add_simulator_c()

labels=[]
data=[]
indexes=[]
v=[]
triangles=[]
for x in model.omega:
    comp=model.simulator.get_comp(x)
    ps=[]
    ws=[]
    for k,v in model.simulator.pd.get_decomposition(Composition(comp)).items():
        if v>0.001:
            ps.append(k)
    #ps=[x if y>0.0001 else None for x,y in zip(ps,ws)]
    if len(ps)==3:
        triangle=[x.composition for x in ps]
        if not triangle in triangles:
            triangles.append(triangle)
    for k in ps:
        label=k.composition
        rep=[k.composition.get_atomic_fraction(Element(x)) for x in
             species_dict]
        rep_con=model.convert_point_to_constrained(rep)
        if label not in labels:
            labels.append(label)
            print(label.reduced_formula)
            x=rep_con[0]
            y=rep_con[1]
            data.append([x/cube_size,y/cube_size,label.reduced_formula])

print(model.convert_point_to_constrained(contained_point))
cube_size=1
cp=np.array(contained_point)/sum(contained_point)
model.setup_from_species_dict(
    species_dict,cube_size=cube_size,basis=basis,contained_point=cp)
model.add_simulator_c()

plot=Square(model)
plot.make_binaries_as_corners()
plot.create_fig(corner_text=False,lw=1)
plot.fig.update_traces(marker_size=0.1,marker_color='black')

#explanation figure
#plot lines
for ts in triangles:
    for n,ta in enumerate(ts):
        a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
        for tb in ts[n+1:]:
            plot_line=True
            b=[tb.get_atomic_fraction(Element(x)) for x in species_dict]
            if str(tb)=='Mg6 B2 O6 F6' or str(ta)=='Mg6 B2 O6 F6':
                pass
                #plot_line=False

            if plot_line:
                plot.plot_line(a,b,w=1,convert=True)
df=pd.DataFrame(columns=['x','y','label'],data=data)

#sample 1
a=[6,2,6,6]
a_con=model.convert_point_to_constrained(a)

b=[5,3,9,1]
b_con=model.convert_point_to_constrained(b)

c=[2,2,5,0]
c_con=model.convert_point_to_constrained(c)

s_con=a_con/3+b_con/3+c_con/3

k_con=b_con/2+c_con/2
k_con2=0.3*b_con+0.7*c_con
model.add_sksig(s_con,k_con,0.1+np.ones((2,2)))
model.add_sksig(s_con,k_con2,0.1+np.ones((2,2)))
model.set_end_points_from_s()
'''

'''
e=model.end_points[0]
plot.plot_line(
    k_con,e,w=1.5,c='green',legend='True estimated direction for Sample 1')
e=model.end_points[1]
plot.plot_line(
    k_con2,e,w=1.5,c='green',
    legend='Measured estimated direction for Sample 1',dash='dash')
plot.plot_point(
    s_con,label='Sample 1',
    s=8,w=1,symbol='133',c='black')
plot.plot_point(
    k_con,label='$\\textrm{True } k \\textrm{ for Sample 1}$',
    s=4,w=1.5,c='black')
plot.plot_point(
    k_con2,label='$\\textrm{Measured } k \\textrm{ for Sample 2}$',
    s=4,w=1.5,c='black')

#sample 2
a=[6,2,6,6]
a_con=model.convert_point_to_constrained(a)

b=[1,0,0,2]
b_con=model.convert_point_to_constrained(b)

c=[1,0,1,0]
c_con=model.convert_point_to_constrained(c)

s_con=a_con/3+b_con/3+c_con/3

k_con=b_con/2+c_con/2
k_con2=0.3*b_con+0.7*c_con
model.add_sksig(s_con,k_con,0.1+np.ones((2,2)))
model.add_sksig(s_con,k_con2,0.1+np.ones((2,2)))
model.set_end_points_from_s()

e=model.end_points[2]
plot.plot_point(
    k_con2,label='$\\textrm{Measured }$ k \\textrm{ for Sample 2}$',
    s=6,w=1.5,c='black')
plot.plot_point(
    k_con,label='$\\textrm{True } k \\textrm{ for Sample 2}$',
    s=6,w=1.5,c='black')
k_con2=0.3*b_con+0.7*c_con
plot.plot_point(
    k_con2,label='$\\textrm{Measured }$ k \\textrm{ for Sample 2}$',
    s=6,w=1.5,c='black')
model.add_sksig(s_con,k_con,0.1+np.ones((2,2)))
model.add_sksig(s_con,k_con2,0.1+np.ones((2,2)))
model.set_end_points_from_s()

e=model.end_points[2]
plot.plot_line(
    k_con,e,w=1.5,c='green',legend='True estimated direction for Sample 2')
e=model.end_points[3]
plot.plot_line(
    k_con2,e,w=1.5,c='green',legend='Measured estimated direction for Sample 2',
    dash='dash')
plot.plot_point(
    s_con,label='Sample 2',
    s=8,w=1,c='black',symbol='133')

plot.custom_scatter(
    df[df['label']=='Mg3B(OF)3'],s=8,c='red',w=1,symbol='134',
    legend='$u: \\textrm{Mg}_3\\textrm{B(OF)}_3$')
plot.custom_scatter(df[df['label']!='Mg3B(OF)3'],s=7,c='black')
lw=1
print(df)
#x=np.array([-0.6,0.6])*cube_size
#y=np.array([-0.1,1.1])*cube_size
y=np.array([-0.07,1.1])*cube_size
x=np.array([-0.63,0.42])*cube_size
plot.fig.update_xaxes(range=x)
plot.fig.update_yaxes(range=y)
    #title_text=("x<sup>'</sup><sub>0</sub> = [<b>B</b><sup>"

plot.custom_scatter(
    df[df['label']=='Mg3B(OF)3'],s=8,c='red',w=1,symbol='134',
    legend='$u: \\textrm{Mg}_3\\textrm{B(OF)}_3$')
plot.custom_scatter(df[df['label']!='Mg3B(OF)3'],s=7,c='black')
lw=1
print(df)
#x=np.array([-0.6,0.6])*cube_size
#y=np.array([-0.1,1.1])*cube_size
y=np.array([-0.07,1.1])*cube_size
x=np.array([-0.63,0.42])*cube_size
plot.fig.update_xaxes(range=x)
plot.fig.update_yaxes(range=y)
plot.fig.update_xaxes(
    #title_text=("x<sup>'</sup><sub>0</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>0</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=8,
    constrain='domain',
    zeroline=False,
    #showticklabels=False,
    #side='bottom',
    tickmode = 'linear',
    tickvals = np.arange(-0.6, 0.41, 0.2),
    tick0=-0.6,
    dtick=0.2,
    ticktext = [f"{i}u" for i in np.arange(-0.6, 0.41, 0.2)]
)
print(np.arange(-0.6,0.41,0.2))
xaxis = dict(
)

plot.fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    #title_text=("x'<sub>1</sub> = [<b>B</b><sup>"
    #            +"T</sup>(<b>x</b>-<b>c</b>)]<sub>1</sub>"
    #            ),
    showline=True,
    linewidth=lw,
    linecolor='black',
    title_font_color='black',
    tickfont_size=8,
    constrain='domain',
    zeroline=False
    #showticklabels=False
)


#plot.fig.update_yaxes(range=[0.35,1.08])
#plot.fig.update_xaxes(range=[-0.56,0.01])
#plot.fig.update_yaxes(range=[-0.05,1.1])
#plot.fig.update_xaxes(range=[-0.6,0.42])
img=('../paper/search',4000,4000)
plot.fig.update_traces(showlegend=False)
plot.show(s=14,show=True)
plot.fig.write_image('../simulation/field_diagrams/linelen2.png',scale=3)
print(model.convert_point_to_constrained(cp))
#plot.fig.update_traces(marker_size=2)
#img=('../paper/plane',4000,4000)
#plotter.show(img=img)


#plot.show(saveimg="../figures/test",save="../figures/test")
print('Done')
'''
