from simulating_phase_fractions import *
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PDPlotter
from fielddata import *
from multiprocessing import Pool
from phasefield import *
from wtconversion import *
import tqdm
import time
import istarmap
import logging, sys
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def run_batch_e(
        species_dict,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us):
    #runs set of simulations for all batch sizes for all stds for a single
    #start point

    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']

    data=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_e(
            species_dict,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    data.append(simulate(copy.deepcopy(model),run_parameters))
    df=pd.DataFrame(data=data,columns=columns)
    return df

def run_batch_c(
        species_dict,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us,plotting=False):
    #runs set of simulations for all batch sizes for all stds for a single
    #start point

    seed=int(time.time())
    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']
    data=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_c(
            species_dict,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    seed+=1
                    data.append(simulate(copy.deepcopy(model),seed,
                                         run_parameters,plotting=plotting))
    df=pd.DataFrame(data=data,columns=columns)
    return df

def run_batch_c_intermetallic(
        phase_field,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us):
    #runs set of simulations for all batch sizes for all stds for a single
    #start point

    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']

    data=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_c_intermetallic(
            phase_field,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    data.append(simulate(copy.deepcopy(model),run_parameters))
    df=pd.DataFrame(data=data,columns=columns)
    return df


def run_batch_parallel_e(
        species_dict,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us,repeats):

    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']

    args=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_e(
            species_dict,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    for i in range(repeats):
                        args.append(
                            (copy.deepcopy(model),
                             copy.deepcopy(run_parameters)))

    data=[]
    with Pool(24) as pool:
        for _ in tqdm.tqdm(
            pool.istarmap(simulate,args),total=len(args)):
            if _ is not None:
                data.append(_)
    df=pd.DataFrame(data=data,columns=columns)
    return df

def run_batch_parallel_c(
        species_dict,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us,repeats):
    seed=int(time.time())
    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']

    args=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_c(
            species_dict,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    for i in range(repeats):
                        seed+=1
                        args.append(
                            (copy.deepcopy(model),seed,
                             copy.deepcopy(run_parameters)))

    data=[]
    '''
    with Pool() as pool:
        for result in pool.starmap(simulate,args):
            data.append(result)
    print(len(args),'lenars')
    '''
    with Pool() as pool:
        for _ in tqdm.tqdm(
            pool.istarmap(simulate,args),total=len(args)):
            if _ is not None:
                data.append(_)
    df=pd.DataFrame(data=data,columns=columns)
    return df

def run_batch_parallel_intermetallic(
        phase_field,mechanistic_parameters,run_parameters,stds,batch_sizes,
        prior_sizes,us,repeats):

    columns=list(run_parameters.keys())
    for i in range(run_parameters['Num. batches']+1):
        columns.append('d(mu,u) for batch' + str(i))
        columns.append('true second moment for batch' + str(i))
        columns.append('d(closest,u) for batch' + str(i))
        columns.append('purity for batch' + str(i))
    columns+=['End type','Predicted error']

    args=[]
    for u in us:
        run_parameters['Unknown']=u
        model=setup_model_c_intermetallic(
            phase_field,mechanistic_parameters,run_parameters['Unknown'])
        for std in stds:
            run_parameters['Experimental std']=std
            for batch_size in batch_sizes:
                run_parameters['Batch size']=batch_size
                for prior_size in prior_sizes:
                    run_parameters['Prior size']=prior_size
                    #logging.debug(f'{std},{batch_size}')
                    for i in range(repeats):
                        args.append(
                            (copy.deepcopy(model),
                             copy.deepcopy(run_parameters)))

    data=[]
    with Pool(24) as pool:
        for _ in tqdm.tqdm(
            pool.istarmap(simulate,args),total=len(args)):
            if _ is not None:
                data.append(_)
    df=pd.DataFrame(data=data,columns=columns)
    return df

def setup_model_e(species_dict,k,u):
    phase_field=list(species_dict.keys())
    e_data=Field_Data(phase_field)
    model=Phase_Field()
    model.set_mechanistic_parameters(k)
    model.unknown_phase=u
    model.setup_from_species_dict(
        species_dict,cube_size=k['cube_size'],
        contained_point=e_data.pure_phases[u],
        show=False)
    model.add_simulator_e(e_data.pure_phases,e_data.triangles)
    model.goal=model.simulator.pure_phases[u]
    model.voided_triangles=[]
    return model

def setup_model_c(species_dict,k,u):
    phase_field=list(species_dict.keys())
    model=Phase_Field()
    model.set_mechanistic_parameters(k)
    model.unknown_phase=u
    comp=Composition(u)
    cp=[comp.get_atomic_fraction(Element(x)) for x in phase_field]
    #DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    #DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
    model.setup_from_species_dict(
        species_dict,cube_size=k['cube_size'],
        contained_point=cp,
        show=True,basis=basis)
    model.add_simulator_c()
    model.goal=model.convert_point_to_constrained(cp)
    model.voided_triangles=[]
    return model

def setup_model_c_intermetallic(phase_field,k,u):
    model=Phase_Field()
    model.set_mechanistic_parameters(k)
    model.unknown_phase=u
    comp=Composition(u)
    cp=[comp.get_atomic_fraction(Element(x)) for x in phase_field]
    model.setup_b(
        phase_field,cube_size=k['cube_size'],contained_point=cp,show=False)
    model.add_simulator_c()
    model.goal=model.convert_point_to_constrained(cp)
    model.voided_triangles=[]
    return model

def setup_plot(plot,model,cube_size,filename='test',lines_etc=True,show=False):
    #find trianngles
    labels=[]
    data=[]
    indexes=[]
    v=[]
    triangles=[]
    for x in model.omega:
        comp=model.simulator.get_comp(x)
        ps=[]
        ws=[]
        for k,v in model.simulator.pd.get_decomposition(
                Composition(comp)).items():
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
                x=rep_con[0]
                y=rep_con[1]
                data.append(
                    [x,y,label.reduced_formula])
    if lines_etc:
        #plot triangle lines
        for ts in triangles:
            for n,ta in enumerate(ts):
                a=[ta.get_atomic_fraction(Element(x)) for x in species_dict]
                for tb in ts[n+1:]:
                    plot_line=True
                    b=[tb.get_atomic_fraction(
                        Element(x)) for x in species_dict]
                    if str(tb)=='Mg2 F4' and str(ta)=='Mg1 O1':
                            plot_line=True
                    if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg2 F4':
                            plot_line=True
                    if str(ta)=='Mg6 B2 O6 F6' and str(tb)=='Mg1 O1':
                            plot_line=True
                    if str(tb)=='Mg6 B2 O6 F6' or str(ta)=='Mg6 B2 O6 F6':
                            plot_line=False

                    if plot_line:
                        plot.plot_line(a,b,w=4,convert=True)
        #plot known phases
        df=pd.DataFrame(columns=['x','y','label'],data=data)
        plot.custom_scatter(df[df['label']!='Mg3B(OF)3'],s=20,c='black')
        plot.custom_scatter(df[df['label']=='Mg3B(OF)3'],s=20,c='red')

    img=('../../paper/h'+filename,1000,1000)
    lw=3
    t=26
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
    plot.fig.update_layout(
        height=pixel_height,width=pixel_width,showlegend=False)
    plot.fig.update_xaxes(#range=x,
                          showline=False,
                          showticklabels=False,
                          linewidth=lw,
                          linecolor='black',
                          zeroline=False,
                          showgrid=False)
    plot.fig.update_yaxes(#range=y,
                          showline=False,
                          #scaleanchor = "x",
                          #scaleratio = 1,
                          showticklabels=False,
                          linewidth=lw,
                          linecolor='black',
                          zeroline=False,
                          showgrid=False)
    if show:
        plot.show()
    else:
        plot.show(img=img,s=t,show=False)


def simulate(model,seed,param_dict,plotting=False,ind=True,method=None):

    use_mean=True
    random.seed(seed)
    np.random.seed(seed)
    if not ind:
        if plotting:
            if method is None:
                pass
            else:
                fig=make_subplots(
                    rows=2,cols=3,
                    shared_yaxes='all',
                    shared_xaxes='all',
                    horizontal_spacing=0.01,
                    vertical_spacing=0.01,
                )
                fig.update_yaxes(
                    scaleanchor = "x",
                    scaleratio = 1,
                )
                fig.update_xaxes(showticklabels=False,autorange=True)
                fig.update_yaxes(showticklabels=False,autorange=True)
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        yanchor="top", y=1,x=0.99,ticks="outside"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
    n=param_dict['Num. batches']
    s=param_dict['Batch size']
    n_p=param_dict['Prior size']
    u=param_dict['Unknown']
    #std=param_dict['Experimental std']
    std=param_dict['Experimental std']
    min_u=param_dict['Min u %']
    pred_error=param_dict['Initial predicted error']
    model.predicted_error=pred_error

    wt_convert=wt_converter(relative=param_dict['Relative error'])
    start=model.random_start_with_u(min_u,size=n_p)

    for point in start:
        try:
            #get mass weights that have an error applied to them from vonmises 
            ps,ws=model.get_errored_mass_weights(
                point,'gaussian_relative+',u,std=std,min_u=min_u,comp=False)
        except OnlyUException:
            #Goal has been found!
            result=[0,0,0,1]*(n+1)
            result+=['Found',model.predicted_error]
            logging.debug('Found at start')
            return list(param_dict.values())+result

        #convert mass weights, predicted weights error to moles, moles error
        predicted_errors=[model.predicted_error for x in ps]
        moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
            ps,ws,phase_field=model.phase_field,weights_error=predicted_errors,
            total_mass=model.total_mass)
        #convert moles and moles error to merged mean and sigma
        k,sig=model.compute_k_sig(moles,formulas_standard,moles_error)
        #add point and make p
        try:
            model.add_sksig(point,k,sig,make_p=True)
        except ZeroPException:
            #model.show('all lines')
            logging.debug(f'Interpolation failure at start: STD {std},'
                          +' num points {len(model.s)}.')
            cd=model.get_score('closest distance')
            purity=model.get_score('purity')
            results=[np.nan]*4*(n+1)
            results+=['Nulled',model.predicted_error]
            return list(param_dict.values())+results
            #store data incase cones need to be thiccened
        model.add_knowns_weights(ps,ws)


    try:
        model.normalise_values(thiccen_cones=False)
    except ConflictingDataException:
        logging.debug(
            f'Nulled start: STD {std},number of points {len(model.s)}.')
        cd=model.get_score('closest distance')
        purity=model.get_score('purity')
        result=[np.nan,np.nan,np.nan,np.nan]*(n+1)
        result+=['Prior was null',model.predicted_error]
        #model.show('all lines')
        return list(param_dict.values())+result
    if np.isnan(model.values).any():
        raise Exception('Nan in model values')
    results=[]
    results.append(model.get_score('d_g_mu'))
    results.append(model.get_score('true_second_moment'))
    results.append(model.get_score('closest distance'))
    results.append(model.get_score('purity'))

    if plotting:
        plotted_points=len(model.s)
        #create fig
        plot=Square(model)
        plot.make_binaries_as_corners()
        plot.create_fig(corner_text=False,lw=2)
        plot.fig.update_traces(marker_size=0.1,marker_color='black')

        #plot line for samples
        model.set_end_points_from_s()
        for sample,e in zip(model.s,model.end_points):
            if n==0:
                plot.plot_line(
                    sample,e,c='green',w=5,
                    legend='Estimated direction')
            else:
                plot.plot_line(sample,e,c='green',w=5)
            plot.plot_point(
                sample,label='Average known',s=20,w=4,symbol='133')
        setup_plot(
            plot,model,model.cube_size,filename='points_0',lines_etc=True,
            show=True)
        plot.plot_point(
            model.goal,label='Average known',c='red',s=20,w=4,symbol='133')
        #fig for p
        plot=Square(model)
        plot.make_binaries_as_corners()
        plot.create_fig(corner_text=False,lw=2)

        #plot.plot_point(model.goal,label='u',s=20,symbol=134,w=4,c='red')
        plot.plot_p(model.omega,model.values,s=4)
        setup_plot(
            plot,model,model.cube_size,filename='p_0',lines_etc=True,
            show=True)

        '''
        if not ind:
            plot.fig.data=plot.fig.data[::-1]
            for i in plot.fig.data:
                fig.add_trace(i,col=1,row=2)
                '''

    for i in range(n):
        p_changed=False
        model.set_next_batch(
            'sample from p',s,p=model.values,use_mean=use_mean)
        #Todo -> implement mean (requires hull testing to work for points 
        #very very close to an edge

        for j,point in enumerate(model.next_batch):
            try:
                ps,ws=model.get_errored_mass_weights(
                    point,'gaussian_relative+',u,std=std,min_u=min_u,comp=False)
            except OnlyUException:
                #Goal has been found!
                results+=[0,0,0,1]*(n-i)
                results+=['Found',model.predicted_error]
                logging.debug(f'Found,{i,j}')
                return list(param_dict.values())+results
            except NoUException:
                #No U in sample
                #TODO (implement a circle of 0 p ?
                pass
            except SmallUException:
                p_changed=True
                if s==1:
                    use_mean=False
            else:
                use_mean=True
                p_changed=True
                predicted_errors=[model.predicted_error for x in ps]
                moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
                    ps,ws,phase_field=model.phase_field,
                    weights_error=predicted_errors,total_mass=model.total_mass)
                k,sig=model.compute_k_sig(moles,formulas_standard,moles_error)
                try:
                    model.add_sksig(point,k,sig,make_p=True)
                except ZeroPException:
                    #model.show('all lines')
                    logging.debug(
                        f'Interpolation failure: STD {std},number of '
                        +'points {len(model.s)}.')
                    cd=model.get_score('closest distance')
                    purity=model.get_score('purity')
                    results+=[results[-4],results[-3],cd,purity]*(n-i)
                    results+=['Nulled',model.predicted_error]
                    return list(param_dict.values())+results
                #store data incase cones need to be thiccened
                model.add_knowns_weights(ps,ws)
                #ps_c=np.array([model.simulator.pure_phases[x] for x in ps])
                #figb,az=plt.subplots(1)
                #model.simulator.show(az)
                #az.scatter(ps_c[:,0],ps_c[:,1])
                #az.scatter([k[0]],[k[1]])
                #print('show')


        if not p_changed:
            #model.show()
            logging.debug(f'Batch {i} was null (Batch size: {s})')
            #None of batch was in simplex with U as vertex
            #logging.debug('Entire batch in null section')
            void_triangles=[]
            for point in model.next_batch:
                t=model.simulator.get_triangle_index(point)
                if not t in void_triangles and not t in model.voided_triangles:
                    void_triangles.append(t)
                    model.voided_triangles.append(t)
            model.omega=model.simulator.filter_omega(
                model.omega,void_triangles)
            try:
                model.recompute_values_full()
            except ZeroPException:
                #model.show('all lines')
                logging.debug(
                    f'Interpolation failure: STD {std},number of '
                    +'points {len(model.s)}. After thicen')
                cd=model.get_score('closest distance')
                purity=model.get_score('purity')
                results+=[results[-4],results[-3],cd,purity]*(n-i)
                results+=['Nulled',model.predicted_error]
                return list(param_dict.values())+results
            model.values=np.prod(model.values_full,axis=0)
            #model.show()

        if np.isnan(model.values).any():
            raise Exception('Nan in model values')
        try:
            model.normalise_values(thiccen_cones=False)
        except ConflictingDataException:
            #model.show('all lines')
            logging.debug(f'Nulled: STD {std},number of points {len(model.s)}.')
            cd=model.get_score('closest distance')
            purity=model.get_score('purity')
            results+=[results[-4],results[-3],cd,purity]*(n-i)
            results+=['Nulled',model.predicted_error]
            return list(param_dict.values())+results
        except ZeroPException:
            #model.show('all lines')
            logging.debug(f'Interpolation failure: STD {std},number of '
                          +'points {len(model.s)}. after normalise')
            cd=model.get_score('closest distance')
            purity=model.get_score('purity')
            results+=[results[-4],results[-3],cd,purity]*(n-i)
            results+=['Nulled',model.predicted_error]
            logging.debug('Nulled')
            return list(param_dict.values())+results
        if np.isnan(model.values).any():
            raise Exception('Nan in model values')
        results.append(model.get_score('d_g_mu'))
        results.append(model.get_score('true_second_moment'))
        results.append(model.get_score('closest distance'))
        results.append(model.get_score('purity'))

        if plotting:
            if i <12:
                #create fig
                plot=Square(model)
                plot.make_binaries_as_corners()
                plot.create_fig(corner_text=False,lw=2)
                plot.fig.update_traces(marker_size=0.1,marker_color='black')

                #plot line for samples
                model.set_end_points_from_s()
                for sample,e in zip(
                        model.s[plotted_points:],model.end_points[plotted_points:]):
                    plot.plot_line(sample,e,c='green',w=5)
                for sample in model.next_batch:
                    plot.plot_point(
                        sample,s=20,w=4,symbol='133')
                setup_plot(
                    plot,model,model.cube_size,filename=f'points_{i+1}',
                    lines_etc=True,show=True)

                #fig for p
                plot=Square(model)
                plot.make_binaries_as_corners()
                plot.create_fig(corner_text=False,lw=2)
                plot.plot_p(model.omega,model.values,s=4)
                setup_plot(
                    plot,model,model.cube_size,filename=f'p_{i+1}',
                    lines_etc=True,show=True)
                plotted_points=len(model.s)
            else:
                print('done')
                input()

            '''
            if i<2:
                #create fig
                plot=Square(model,method='plt')
                plot.make_binaries_as_corners()
                plot.create_fig()
                #plot goal and all sample points
                plot.plot_points_np([model.goal],s=60,zorder=5,m='+')
                plot.plot_points_np(
                    model.next_batch,s=60,c='blue',labels=range(len(start)),
                    m='2')
                plot.plot_points_np([model.get_mean()],s=60,c='green',m='x',zorder=4)
                #plot_triangulation
                pd=model.simulator.pd
                matplotter=PDPlotter(pd)
                for line in matplotter.lines:
                    entry1=pd.qhull_entries[line[0]]
                    entry2=pd.qhull_entries[line[1]]
                    entry1_s=[
                        entry1.composition.get_atomic_fraction(Element(x)) for x in 
                        model.phase_field]
                    entry2_s=[
                        entry2.composition.get_atomic_fraction(Element(x)) for x in 
                        model.phase_field]
                    if (abs(np.dot(entry2_s,model.normal_vectors[0]))<1e-5 and
                            abs(np.dot(entry1_s,model.normal_vectors[0]))<1e-5):
                        a=model.convert_point_to_constrained(entry1_s)
                        b=model.convert_point_to_constrained(entry2_s)
                        plot.plot_line(a,b,w=0.5)


                #plot line for samples
                model.set_end_points_from_s()
                for sample,e in zip(
                        model.s[plotted_points:],model.end_points[plotted_points:]):
                    plot.plot_line(sample,e,c='blue',w=0.8)
                plotted_points=len(model.s)
                plot.show(save=str(2*i+2))
                if not ind:
                    for t in plot.fig.data:
                        fig.add_trace(t,col=i+2,row=1)
                #fig for p
                plot=Square(model,method='plt')
                plot.make_binaries_as_corners()
                plot.create_fig()
                #x=np.log(model.values)
                #x[np.isneginf(x)]=-999
                x=model.values
                plot.plot_p(model.omega,model.values)
                plot.show(save=str(2*i+3))
                if not ind:
                    plot.fig.data=plot.fig.data[::-1]
                    for t in plot.fig.data:
                        fig.add_trace(t,col=i+2,row=2)
            else:
                print('END')
                #fig.show()
                input()
                '''


    results.append('Terminated successfully')
    logging.debug('terminated successfully')
    results.append(model.predicted_error)
    #logging.debug('Finished')
    return list(param_dict.values())+results

if __name__ == "__main__":
    #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stderr, level=None)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True


    #constant hyperparams
    param_dict={
        'Num. batches':15,
        'Min u %':0.1,
        'p_e':0.02,
        'Experimental std':None,
        'Batch size':None,
        'Prior size':None,
        'Unknown':None,
        'Relative error':False,
    }
    mechanistic_dict={
        'cube_size':100,
        'use_fib':True,
        'demi_len':100,
        'line_len':50,
        'total_mass':1,
    }

    #variable Sigma_E Sigma_P and Batch size
    stds=[0.03] #sigma E
    p_e=[0.1] #
    batch_sizes=[1]
    prior_sizes=[1]

    repeats=50
    plotting=True

    #specific data for phase fields

    #MgBOF phase_field
    species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
    #us=["Mg 6 B 2 O 6 F 6","Mg 20 B 12 O 36 F 4"]
    us=["Mg 6 B 2 O 6 F 6"]
    filename='../simulation/data/testing/MgBOF_plots.csv'
    #MgAlCu  
    '''
    phase_field=['Mg','Al','Cu']
    us=['Mg 6 Al 4 Cu 8','Mg 2 Al 4 Cu 2','Mg 6 Al 15 Cu 18']
    filename='../simulation/testing/asdf.csv'
    '''
    #LiAlBO
    '''
    species_dict={'Li':1,'Al':3,'B':3,'O':-2}
    us=["Li 2 Al B 5 O 10",
        "Li Al B 2 O 5",
        "Li 3 Al B 2 O 6",
        "Li 2 Al B O 4",
        "Li Al 7 B 4 O 17",
        "Li 2.46 Al 0.18 B O 3"]
    filename='../simulation/hyperparameters/LiAlBO.csv'
    '''

    #run intermetallic
    '''
    for i in range(repeats):
        df=run_batch_c_intermetallic(
            phase_field,mechanistic_dict,param_dict,stds,batch_sizes,
            prior_sizes,us)
        with open(filename, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
            '''
    #run parallel intermetallic 
    '''
    for i in range(99999999999):
        df=run_batch_parallel_intermetallic(
            phase_field,mechanistic_dict,param_dict,stds,batch_sizes,
            prior_sizes,us,repeats)
        with open(filename, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
            '''
    #run computed
    for i in range(repeats):
        for e in p_e:
            param_dict['Initial predicted error']=e
            df=run_batch_c(
                species_dict,mechanistic_dict,param_dict,stds,batch_sizes,
                prior_sizes,us,plotting=plotting)
        with open(filename, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
    #run computed parallel
    '''
    for i in range(60):
        print(i)
        for e in p_e:
            param_dict['Initial predicted error']=e
            df=run_batch_parallel_c(
                species_dict,mechanistic_dict,param_dict,stds,batch_sizes,
                prior_sizes,us,repeats)
            with open(filename, 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)
                '''
    #run experimental
    '''
    for i in range(repeats):
        df=run_batch_e(
            species_dict,mechanistic_dict,param_dict,stds,batch_sizes,
            prior_sizes,us)
        with open(filename, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
            '''
    #run experimental parallel
    '''
    for i in range(100):
        for cube_size in [50,100,200]:
            for points_per_line in [50,100,200]:
                print(points_per_line)
                for num_lines in [50,100,200]:
                    mechanistic_dict['demi_len']=num_lines
                    mechanistic_dict['cube_size']=cube_size
                    mechanistic_dict['line_len']=points_per_line
                    df=run_batch_parallel_e(
                        species_dict,mechanistic_dict,param_dict,stds,batch_sizes,
                        prior_sizes,us,repeats)
                    df['demi_len']=num_lines
                    df['cube_size']=cube_size
                    df['line_len']=points_per_line
                    with open(filename, 'a') as f:
                        df.to_csv(f, mode='a', header=f.tell()==0)
                        '''

    #for u in us:
        #df=run_batch_parallel(
            #batch_sizes,stds,u,predicted_error,num_batches,repeats,min_us)
        #df=run_batch(batch_sizes,stds,u,predicted_error,num_batches,repeats,min_us)
    #with open(filename, 'a') as f:
    #    df.to_csv(f, mode='a', header=f.tell()==0)
    '''
    for i in range(999999999):
            try:
                df=run_batch_parallel(
                    batch_sizes,stds,u,predicted_error,num_batches,repeats,
                    min_us)
                #df=run_batch(batch_sizes,stds,u,predicted_error,num_batches,repeats)
                with open(filename, 'a') as f:
                    df.to_csv(f, mode='a', header=f.tell()==0)
            except():
                print('Error')
                '''

