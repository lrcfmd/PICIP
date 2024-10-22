import numpy as np
from phasefield import *
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
import math

class Square():
    #plotting fun ction for when constrained space is two dimensional
    
    def __init__(self,model,method='plotly'):
        self.model=model
        self.cube_size=model.cube_size
        self.elements=model.phase_field
        self.method=method

    def add_corners(self,corners,labels=None):
        #corners: a list of lists of floats giving composition for each
        #corner in standard representation
        #labels: corresponding corner labels
        self.corners=corners
        corners=self.model.convert_point_to_constrained(corners)
        if labels is None:
            labels=self.get_composition_string_standard(corners,zeros=False)
        df=pd.DataFrame(corners,columns=['x','y'])
        df['Label']=labels
        self.corners_df=df
        self.xmin=df['x'].min()
        self.xmax=df['x'].max()
        self.ymin=df['y'].min()
        self.ymax=df['y'].max()
        self.labels=labels

    def make_binaries_as_corners(self,automated_text_position=True):
        charges=self.model.normal_vectors[0]
        try:
            charges=charges.astype(int)
        except:
            raise Exception(
                'Could not convert charges to integer')
        corners=[]
        labels=[]
        if np.all(charges[:2]>0) and np.all(charges[2:]<0):
            for n,cation in enumerate(charges[:2]):
                for m,anion in enumerate(charges[2:]):
                    anion=-1*anion
                    lcm=math.lcm(cation,anion)
                    cation_co=lcm/cation
                    anion_co=lcm/anion
                    corner=[0]*4
                    corner[n]=cation_co
                    corner[m+2]=anion_co
                    labels.append(
                        self.get_composition_string_standard(
                            corner,zeros=False))
                    corners.append(corner)
        elif np.all(charges[:3]>0) and charges[3]<0:
            anion=-charges[3]
            for n,cation in enumerate(charges[:3]):
                    lcm=math.lcm(cation,anion)
                    cation_co=lcm/cation
                    anion_co=lcm/anion
                    corner=[0]*4
                    corner[n]=cation_co
                    corner[3]=anion_co
                    labels.append(
                        self.get_composition_string_standard(
                            corner,zeros=False))
                    corners.append(corner)
        elif charges[0]>0 and np.all(charges[1:]<0):
            cation=charges[0]
            for m,anion in enumerate(charges[1:]):
                    anion=-1*anion
                    lcm=math.lcm(cation,anion)
                    cation_co=lcm/cation
                    anion_co=lcm/anion
                    corner=[0]*4
                    corner[0]=cation_co
                    corner[m+1]=anion_co
                    labels.append(
                        self.get_composition_string_standard(
                            corner,zeros=False))
                    corners.append(corner)
        else:
            raise Exception(
                'Automatic corner generation only implemented for'
                +'single formal charge phase fields ordered with positive'
                +'species first.')
        self.add_corners(corners,labels)
        if automated_text_position:
            self.set_corner_text_position(corners)

    def set_corner_text_position(self,corners):
        text_position=[]
        for x in corners:
            x_c=self.model.convert_point_to_constrained(x)
            x_x=None
            if abs(x_c[0]-self.xmin)<abs(x_c[0]-self.xmax):
                x_x='left'
            else:
                x_x='right'
            x_y=None
            if abs(x_c[1]-self.ymin)<abs(x_c[1]-self.ymax):
                x_y='bottom'
            else:
                x_y='top'
            if x_c[0]==self.xmin or x_c[0]==self.xmax:
                if x_c[1]!=self.ymin and x_c[1]!=self.ymax:
                    x_y='middle'
            if x_c[1]==self.ymin or x_c[1]==self.ymax:
                if x_c[0]!=self.xmin and x_c[0]!=self.xmax:
                    x_x='center'
            text_position.append(x_y+" "+x_x)
        self.corner_text_position=text_position

    def create_fig(
            self,corner_size=5,text_position='top center',corner_text=True,
            corners=True,phase_map=None,lw=1,size=None):

        if corners:
            #remove x,y from hover_data
            hover_data={}
            for i in self.corners_df.columns:
                hover_data[i]=False
            #conditionally remove corner labels
            if not corner_text:
                self.corners_df['Label']=''
            #plot corners
            if self.method=='plotly':
                self.fig=px.scatter(
                    self.corners_df,x='x',y='y',text='Label',hover_data=hover_data)
                if hasattr(self,'corner_text_position'):
                    text_position=self.corner_text_position
                self.fig.update_traces(marker={'size':corner_size},
                                      textposition=text_position)
            else:
                self.fig,self.ax=plt.subplots()
            #plot lines between corners
            if len(self.corners)==3:
                for n in range(len(self.corners_df)):
                    for m in range(n+1,len(self.corners_df)):
                        df=self.corners_df.iloc[[n,m],:]
                        p=df[['x','y']].to_numpy()
                        self.plot_line(p[0],p[1],w=1.5)
            else:
                for n,(la,ca) in  enumerate(zip(self.labels,self.corners)):
                    indices_a=[i for i,e in enumerate(ca) if e!=0 ]
                    for m,(lb,cb) in enumerate(
                            zip(self.labels[n+1:],self.corners[n+1:])):
                        plot=True
                        indices_b=[i for i,e in enumerate(cb) if e != 0]
                        if bool(set(indices_a) & set(indices_b)):
                            df=self.corners_df.iloc[[n,n+1+m],:]
                            if self.method=='plotly':
                                p=df[['x','y']].to_numpy()
                                if la == 'MgO' and la == 'MgF<sub>2</sub>':
                                    plot=True
                                if plot:
                                    self.plot_line(p[0],p[1],w=lw)
                            else:
                                self.ax.plot(df['x'],df['y'],c='black')
        else:
            self.fig=go.Figure()
        if phase_map is not None:
            d={}
            for k,v in phase_map.items():
                d[k]=np.array(v)/sum(v)
            self.phase_map=d


    def create_fig_corners(self,corners,edges,s=5,norm=100):
        corners_c=self.model.convert_point_to_constrained(corners)
        labels=self.get_composition_string(
            corners_c,zeros=False,norm=norm)
        columns=['x','y']
        df=pd.DataFrame(data=corners_c,columns=columns)
        self.xmin=df['x'].min()
        self.xmax=df['x'].max()
        self.ymin=df['y'].min()
        self.ymax=df['y'].max()
        self.set_corner_text_position(corners)
        df['Label']=labels
        hover_data={}
        for i in df.columns:
            hover_data[i]=False
        self.fig=px.scatter(df,x='x',y='y',text='Label',hover_data=hover_data)
        self.fig.update_traces(marker={'size':s},
                              textposition=self.corner_text_position)
        for edge in edges:
            x=[corners_c[edge[0]][0],corners_c[edge[1]][0]]
            y=[corners_c[edge[0]][1],corners_c[edge[1]][1]]
            fig1 = go.Figure(data=go.Scatter(
                x=x,y=y,mode='lines',line_color='black',hoverinfo='skip',
                showlegend=False))
            self.fig=go.Figure(data=self.fig.data+fig1.data)

    def show(
            self,title=None,show=True,save=None,saveimg=None,legend_title=None,
            s=20,img=None,transparent=False):
        if self.method=='plotly':
            #self.fig.update_xaxes(autorange=True)
            #self.fig.update_yaxes(autorange=True)
            self.fig.update_layout(
                font_size=s,
                coloraxis_showscale=False,
                hoverlabel_font_size=20,
                legend_title=legend_title,
                #height=1000,
                #coloraxis_colorbar=dict(yanchor="top", y=1, x=0,ticks="outside"),
                plot_bgcolor='rgb(255,255,255)',
                paper_bgcolor='rgb(255,255,255)',
                font_family="DejaVu Sans",
                font_color='black',
                #xaxis_range=[self.xmin-self.cube_size/3,self.xmax+self.cube_size/3]
                #yaxis_range=[15,self.xmax+self.cube_size/3]
            )
            if transparent:
                self.fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )

            #self.fig.update_yaxes(
            #    scaleanchor = "x",
            #    scaleratio = 1,
            #)
            if title is not None:
                self.fig.update_layout(title=dict(text=title))
            #self.fig.update_layout(showlegend=False)
            if show:
                self.fig.show()
            if saveimg is not None:
                self.fig.write_image(save+".png")
            if save is not None:
                self.fig.write_html(save+".html")
            #self.fig.data = self.fig.data[::-1]
            if img is not None:
                '''
                pio.write_image(
                    self.fig,img[0]+'.png',format='png',width=img[1],
                    height=img[2],scale=1)
                    '''
                self.fig.update_layout(
                    margin=dict(l=40, r=40, b=40, t=40))  # Adjust margins as needed
                pio.write_image(
                    self.fig,img[0]+'.png',format='png',scale=3)
        else:
            plt.legend()
            plt.savefig("../../paper/PCSb_" + save + ".png")
            plt.show()

    def add_constraint(self,c1,c2,c3,c4=None,label=None):
        #add a custom linear constraint
        #i.e. for MgSiSI to be >30$ Mg: 
        #0.7Mg-0.3Si-0.3S-0.3I=0
        #->c1=0.7, c2,c3,c4=-0.03
        #This constraint is then transformed to constrained basis
        #and saved as a,b,c (ax+by+c=0)
        basis=self.basis
        print('TODO: could this be a function passed by algorithm?')
        cp=self.model.contained_point
        if len(basis[0])==3:
            a=c1*basis[0,0]+c2*basis[0,1]+c3*basis[0,2]
            b=c1*basis[1,0]+c2*basis[1,1]+c3*basis[1,2]
            c=c1*cp[0]+c2*cp[1]+c3*cp[2]
        elif len(basis[0])==4:
            a=c1*basis[0,0]+c2*basis[0,1]+c3*basis[0,2]+c4*basis[0,3]
            b=c1*basis[1,0]+c2*basis[1,1]+c3*basis[1,2]+c4*basis[1,3]
            c=c1*cp[0]+c2*cp[1]+c3*cp[2]+c4*cp[3]

        else:
            print('Error,bad basis length')
        self.lines.append([a,b,c])
        self.line_labels.append(label)

    def plot_constraints(self):
        #plot any consraints stored in self.lines
        x=np.linspace(self.xmin,self.xmax,100)
        xs=[]
        ys=[]
        for i in self.lines:
            y=(-1*i[2]-i[0]*x)/i[1]
            delete=np.where(np.logical_and(y>=self.ymin, y<=self.ymax))
            xs.append(x[delete])
            ys.append(y[delete])
        for x,y,l in zip(xs,ys,self.line_labels):
            self.fig.add_trace(
                go.Scatter(x=x,y=y,hovertemplate=l,showlegend=False))

    def plot_grid(self,size=2,cut=True):
        #plot the discrestised mesh satisfying constraints
        grid=self.model.omega
        if hasattr(self,'lines'):
            for line in self.lines:
                grid=np.delete(
                    grid,np.where(
                        np.einsum('...i,i->...',grid,line[:-1])<-1*line[-1]),
                    axis=0)
        comps=self.get_composition_string(grid)
        self.fig.add_trace(go.Scatter(
            x=grid[:,0],y=grid[:,1],hovertemplate=comps,mode='markers',
            name='Grid'))

    def get_composition_string_old(self,pos,norm=100,zeros=True):
        #function to return list of composition string for each constrained
        #point in array pos
        standard_rep=[]
        for p in pos:
            p_s=self.model.convert_to_standard_basis(p)
            p_s=norm*p_s/sum(p_s)
            standard_rep.append(list(p_s))
        return self.get_composition_string_standard(standard_rep,zeros=zeros)

    def get_composition_string(
            self,pos,form='html',zeros=True,tol=1e-3,corner_attempt=False,
            precision=1,norm=100,lcm=False):
        #function to return list of composition string for each constrained
        #point in array pos
        comps=[]
        for p in pos:
            found=False
            p_s=self.model.convert_to_standard_basis(p,norm=1)
            if hasattr(self,'phase_map'):
                for k,v in self.phase_map.items():
                    if np.allclose(p_s,v):
                        comps.append(k)
                        found=True
            if not found:
                p_s=norm*p_s/sum(p_s)
                if corner_attempt:
                    lcm=np.min(p_s[p_s>tol])
                    p_s=p_s/lcm
                    precision=2
                comp=""
                for x,l in zip(p_s,self.elements):
                    if x>=x.round()-tol and x<=x.round()+tol:
                        x_rep=str(int(x.round()))
                    else:
                        x_rep=str(x.round(precision))
                    if (not zeros and abs(x)>tol) or zeros:
                        if form == 'html':
                            comp+=l
                            if x>1+tol or x<1-tol:
                                comp+="<sub>"+x_rep+"</sub>"
                        elif form =='txt':
                            comp+=l+str(int(x.round()))
                        else:
                            print('error, unknown form')
                comps.append(comp)
        return comps

    def get_composition_string_standard(self,pos,zeros=True,tol=1e-4):
        if not hasattr(pos[0],'__iter__'):
            pos=[pos]
        #function to return list of composition string for each standard
        #point in array pos
        comps=[]
        for p in pos:
            comp=""
            for x,l in zip(p,self.elements):
                if (not zeros and abs(x)>tol) or zeros:
                    comp+=l
                    if x!=1:
                        x_str=None
                        if type(x)==int:
                            x_str=str(x)
                        else:
                            x_str=str(int(x.round()))
                        comp+="<sub>"+x_str+"</sub>"
            comps.append(comp)
        if len(comps)==1:
            return comps[0]
        else:
            return comps

    def plot_triangle_line(self,triangle):
        #plots lines of a simplex
        triangle=self.model.convert_point_to_constrained(triangle)
        for n,i in enumerate(triangle):
            for j in triangle[n+1:]:
                fig=px.line(x=[i[0],j[0]],y=[i[1],j[1]])
                self.fig=go.Figure(data=self.fig.data+fig.data)

    def plot_point(self,point,label=None,c='black',s=5,t='',w=3,symbol=None):
        #plot a point
        fig=go.Figure(go.Scatter(
                x=[point[0]],y=[point[1]],mode='markers+text',
                hovertemplate=label,marker={'color':c,'size':s},text=t))
        if symbol is not None:
            fig.update_traces(marker=dict(symbol=symbol,
                                          line=dict(color='black',width=w)))
        if label is not None:
            fig.update_traces(name=label,showlegend=True)

        self.fig=go.Figure(data=self.fig.data+fig.data)
        



    def plot_line(
            self,start,end,c='black',w=2,legend=None,convert=False,dash=None):
        if convert:
            start=self.model.convert_point_to_constrained(start)
            end=self.model.convert_point_to_constrained(end)
        #plot a line
        if legend is None:
            show_l=False
        else:
            show_l=True
        if self.method=='plotly':
            self.fig.add_trace(
                go.Scatter(
                    x=[start[0],end[0]],y=[start[1],end[1]],mode='lines',
                    line={'color':c,'width':w,'dash':dash},showlegend=show_l,
                    name=legend))
        else:
            self.ax.plot(
                [start[0],end[0]],[start[1],end[1]],c=c,linewidth=w)

    def plot_p(self,omega,values,s=0.5):
        #plot a heatmap (as a coloured scatter)
        if self.method=='plotly':
            df=pd.DataFrame(columns=['x','y'],data=omega)
            df['C']=values
            hover_template={
                'x':False,
                'y':False,
                'C':False}
            fig=go.Figure(px.scatter(df,x='x',y='y',color='C',hover_data=hover_template))
            fig.update_traces(marker_size=s)
            self.fig=go.Figure(data=self.fig.data+fig.data)
        else:
            valuesp=values**0.3
            valuesp=values/sum(values)
            self.ax.scatter(
                omega[:,0],omega[:,1],c=valuesp**0.3,s=0.1,cmap='binary')

    def custom_scatter(
            self,df,w=3,s=30,c=None,composition_text=False,symbol=None,
            legend=None,**kwargs):
        #plot scatter of points in df with additional column information
        #displayed on hover
        df=copy.deepcopy(df)
        hover_data={}
        for i in df.columns:
            if i!='x' and i!='y':
                hover_data[i]=True
            else:
                hover_data[i]=False
        batch=df[['x','y']].to_numpy()
        label=self.get_composition_string(
            batch,norm=3,corner_attempt=True,zeros=False)
        df['Composition']=label
        if composition_text:
            fig=px.scatter(
                df,x='x',y='y',text='Composition',hover_data=hover_data,**kwargs)
        else:
            fig=px.scatter(
                df,x='x',y='y',hover_data=hover_data,**kwargs)
            print(df)

        fig.update_traces(marker_size=s)
        if symbol is not None:
            fig.update_traces(marker=dict(symbol=symbol,
                                          line=dict(color='red',width=w)))

        if c is not None:
            fig.update_traces(marker_color=c)
        if legend is not None:
            fig.update_traces(name=legend,showlegend=True)
        self.fig=go.Figure(data=self.fig.data+fig.data)

    def plot_points(
            self,point_dict=None,file=None,df=None,label=None,compositions=True,
            c=None,s=None,file_format='standard',showlegend=False,symbol=None):
        if point_dict is not None and file is not None:
            raise Exception('Please provide only one of {point_dict,file}')
        if file is not None:
            self.model.add_points_only_from_file(
                file,format_type=file_format)
            name=file
        if point_dict is not None:
            self.model.add_points_from_dict(point_dict)
            name=list(point_dict.keys())[0]
        points=self.model.points
        df=pd.DataFrame(data=points,columns=['x','y'])
        hover_data={'x':False,'y':False}
        if compositions:
            df['Comp']=self.get_composition_string(points)
            hover_data['Comp']=True
        if label is not None:
            df['Label']=self.model.get_points_labels(label)
            hover_data['Label']=True
        fig1=px.scatter(
            df,x='x',y='y',hover_data=hover_data)
        fig1.update_traces(name=name,showlegend=showlegend)
        if c is not None:
            fig1.update_traces(marker_color=c)
        if s is not None:
            fig1.update_traces(marker_size=s)
        self.fig=go.Figure(data=self.fig.data+fig1.data)
        if symbol is not None:
            fig.update_traces(marker=dict(symbol=symbol,
                                          line=dict(color='black',width=15)))

    def plot_df(self,df,columns=[],colour_by=None,s=5,c='black'):
        if not {'x','y'}.issubset(df.columns):
            raise Exception('Dataframe must have x,y columns')
        if 'z' in df.columns:
            raise Exception('Dataframe contains z but this is a 2d plotter?')

        hover_data={'x':False,
                    'y':False,
                   }
        for i in df.columns:
            if i in columns:
                hover_data[i]=True
            else:
                hover_data[i]=False
        if colour_by is not None:
            dfa=df[~df[colour_by].isna()]
            dfb=df[df[colour_by].isna()]
            fig1 = px.scatter(
                dfa,x='x',y='y',hover_data=hover_data,symbol='Source',
                color=colour_by)
            fig2 = px.scatter(
                dfb,x='x',y='y',hover_data=hover_data,symbol='Source')
            fig2.update_traces(marker_color=c)
            fig2.update_traces(marker_size=s)
            fig1.update_traces(marker_size=s)
            self.fig=go.Figure(data=self.fig.data+fig1.data + fig2.data)
        else:
            fig1 = px.scatter(
                df,x='x',y='y',hover_data=hover_data,symbol='Source')
            fig1.update_traces(marker_size=s)
            fig1.update_traces(marker_color=c)
            self.fig=go.Figure(data=self.fig.data+fig1.data)

    def plot_points_constrained_cell(self,min_a,max_a,save=None):
        points,labels=self.model.get_comps_constrained_cell_labels(min_a,max_a)
        self.plot_points_np(
            points,legend='All allowed compositions',labels=labels,
            composition=False)
        if save is not None:
            points=self.model.convert_to_standard_basis(points,norm=min_a)
            df=pd.DataFrame(data=points,columns=self.elements)
            df.round().astype(int).to_csv(save,index=False)

    def plot_points_np(
            self,batch,labels=None,composition=True,s=10,c=None,legend=None,
            norm=100,m=None,zorder=1,w=4):
        df=pd.DataFrame(data=batch,columns=['x','y'])
        hover_data={'x':False,
                    'y':False,
                   }
        if labels is not None:
            df['Label']=labels
            hover_data['Label']=True
        if composition:
            df['Composition']=self.get_composition_string(batch,norm=norm)
            hover_data['Composition']=True
        if self.method=='plotly':
            fig1 = px.scatter(
                df,x='x',y='y',hover_data=hover_data,)
            fig1.update_traces(marker_size=s)
            if c is not None:
                fig1.update_traces(marker_color=c)
            if legend is not None:
                fig1.update_traces(
                    name=legend,
                    showlegend=True)
            if m is not None:
                fig1.update_traces(marker=dict(symbol=m,
                                          line_width=w))
            self.fig=go.Figure(data=self.fig.data+fig1.data)
        else:
            batch=np.array(batch)
            self.ax.scatter(batch[:,0],batch[:,1],label=legend,c=c,marker=m,zorder=zorder,s=s)

    def plot_text(self,batch,text,pos):
        df=pd.DataFrame(data=batch,columns=['x','y'])
        hover_data={'x':False,
                    'y':False,
                   }
        '''
            fig1 = px.scatter(df,x='x',y='y',hover_data=hover_data)
            fig1.update_traces(marker_size=s)
            fig1.update_traces(marker_color=c)
            if legend is not None:
                fig1.update_traces(
                    name=legend,
                    showlegend=True)
            self.fig=go.Figure(data=self.fig.data+fig1.data)
        else:
            batch=np.array(batch)
            self.ax.scatter(batch[:,0],batch[:,1],label=legend,c=c,marker=m,zorder=zorder,s=s)
            '''


    def plot_computed_corners(self,corners,edges,s=10):
        corners_df=pd.DataFrame()
        corners_df['x']=corners[:,0]
        corners_df['y']=corners[:,1]
        corners_df['Label']=self.get_composition_string(
            corners,zeros=False,corner_attempt=True)
        self.xmin=corners_df['x'].min()
        self.xmax=corners_df['x'].max()
        self.ymin=corners_df['y'].min()
        self.ymax=corners_df['y'].max()
        fig1=px.scatter(
            corners_df,x='x',y='y',text='Label',
            hover_data={'x':False,'y':False,'Label':False})
        fig1.update_traces(marker={'size': s})
        self.fig=go.Figure(data=self.fig.data+fig1.data)
        for edge in edges:
            x=[corners[edge[0]][0],corners[edge[1]][0]]
            y=[corners[edge[0]][1],corners[edge[1]][1]]
            fig1 = go.Figure(data=go.Scatter(
                x=x, y=y, mode='lines',line_color='black',
                hoverinfo='skip',showlegend=False))
            self.fig=go.Figure(data=self.fig.data+fig1.data)

    def plot_model_points_df(
            self,columns=[],colour_by=None,s=5,c='black',composition=True,
            composition_norm=100,symbol=None,legend=''):
        df=self.model.points_df
        hover_data={'x':False,
                    'y':False,
                   }
        for i in df.columns:
            if i in columns:
                hover_data[i]=True
            else:
                hover_data[i]=False
        if composition:
            df['Composition']=self.get_composition_string(
                df[['x','y']].to_numpy(),norm=composition_norm)
            hover_data['Composition']=True
        if colour_by is not None:
            fig1 = px.scatter(
                df,x='x',y='y',hover_data=hover_data,symbol=symbol,
                color=colour_by)
            if symbol is None:
                fig1.update_traces(name=legend,showlegend=True)
            fig1.update_traces(marker_size=s)
            fig1.update_coloraxes(colorbar_title_text=colour_by)
            fig1.update_coloraxes(colorbar_thickness=30)
            self.fig=go.Figure(data=self.fig.data+fig1.data)
        else:
            fig1 = px.scatter(
                df,x='x',y='y',hover_data=hover_data,symbol='Source')
            fig1.update_traces(marker_size=s)
            fig1.update_traces(marker_color=c)
            if symbol is None:
                fig1.update_traces(name=legend,showlegend=True)
            self.fig=go.Figure(data=self.fig.data+fig1.data)



