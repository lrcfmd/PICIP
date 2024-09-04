#from algorithm import all_information
#from visualise_square import *
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from wtconversion import *
import numpy as np
import pickle
import copy
import logging, sys
import warnings
from scipy.optimize import linprog
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from exceptions import *
from scipy.spatial import Delaunay
all_phases=[]
counts=[0]*40
class PhaseFractionSimulator:

    def __init__(self,pure_phases,triangles):
        self.pure_phases=pure_phases
        self.triangles=triangles
        self.max_purity=0

    def get_known_weights(self,p,method,u,**kwargs):
        ps,ws=self.get_errored_mass_weights(p,method,**kwargs)
        if len(ws)==1:
            if ps[0]==u:
                logging.debug('Tested goal!!')
                return 'Success'
        if u in ps:
            i=ps.index(u)
            ps.remove(u)
            ws=np.delete(ws,i)
            if sum(ws)==0:
                print(len(ws))

                ws[np.random.randint(0,len(ws))]=1
            ws=ws/sum(ws)
            return ps,ws
        else:
            logging.debug('Warning sampled point did not contain u')
            return(None,None)

    def get_errored_mass_weights(self,p,method,u,**kwargs):
        phases,weights=self.get_mass_weights(p)
        if len(phases)==3:
            if not phases in all_phases:
                all_phases.append(copy.deepcopy(phases))
            counts[all_phases.index(phases)]+=1
        #print(u,":",phaases)
        weights=np.array(weights)/sum(weights)
        r_weights=[]
        if method=='gaussian':
            if u in phases:
                i=phases.index(u)
                if kwargs.get('min_u'):
                    weights=np.array(weights)/sum(weights)
                    if weights[i]<kwargs['min_u']:
                        raise SmallUException(str(weights[i]))
                if weights[i]>self.max_purity:
                    self.max_purity=weights[i]
                if len(phases)==1:
                    raise OnlyUException()
                phases.remove(u)
                weights=np.delete(weights,i)
                weights=weights/sum(weights)
                if len(phases)==1:
                    logging.debug('Convex hull was goal and only 1 other')
                if len(phases)==2:
                    std=kwargs['std']
                    looking=True
                    while looking:
                        p0 = np.random.normal(loc=weights[0],scale=std)
                        if p0>=0 and p0<=1:
                            looking=False
                    weights=[p0,1-p0]
                return phases,weights
            else:
                raise NoUException()
        elif method=='gaussian_relative':
            if u in phases:
                i=phases.index(u)
                weights=np.array(weights)/sum(weights)
                u_weight=weights[i]
                if kwargs.get('min_u'):
                    if weights[i]<kwargs['min_u']:
                        raise SmallUException(str(weights[i]))
                if weights[i]>self.max_purity:
                    self.max_purity=weights[i]
                if len(phases)==1:
                    raise OnlyUException()
                phases.remove(u)
                weights=np.delete(weights,i)
                weights=weights/sum(weights)
                if len(phases)==1:
                    logging.debug('Convex hull was goal and only 1 other')
                if len(phases)==2:
                    std=kwargs['std']
                    std=std*u_weight
                    looking=True
                    while looking:
                        p0 = np.random.normal(loc=weights[0],scale=std)
                        if p0>=0 and p0<=1:
                            looking=False
                    weights=[p0,1-p0]
                return phases,weights
        elif method=='gaussian_relative+':
            if u in phases:
                i=phases.index(u)
                weights=np.array(weights)/sum(weights)
                u_weight=weights[i]
                if kwargs.get('min_u'):
                    if weights[i]<kwargs['min_u']:
                        raise SmallUException(str(weights[i]))
                if weights[i]>self.max_purity:
                    self.max_purity=weights[i]
                if len(phases)==1:
                    raise OnlyUException()
                phases.remove(u)
                weights=np.delete(weights,i)
                weights=weights/sum(weights)
                if len(phases)==1:
                    logging.debug('Convex hull was goal and only 1 other')
                if len(phases)==2:
                    std=kwargs['std']
                    std=std+std*u_weight
                    looking=True
                    while looking:
                        p0 = np.random.normal(loc=weights[0],scale=std)
                        if p0>=0 and p0<=1:
                            looking=False
                    weights=[p0,1-p0]
                return phases,weights
            else:
                raise NoUException()
        elif method=='vonmises':
            if u in phases:
                i=phases.index(u)
                if kwargs.get('min_u'):
                    weights=np.array(weights)/sum(weights)
                    if weights[i]<kwargs['min_u']:
                        raise SmallUException(str(weights[i]))
                if weights[i]>self.max_purity:
                    self.max_purity=weights[i]
                if len(phases)==1:
                    raise OnlyUException()
                phases.remove(u)
                weights=np.delete(weights,i)
                weights=weights/sum(weights)
                if len(phases)==1:
                    logging.debug('Convex hull was goal and only 1 other')
                if len(phases)==2:
                    mu=np.arctan(weights[0]/weights[1])
                    kappa=kwargs['kappa']
                    looking=True
                    while looking:
                        theta = vonmises.rvs(kappa, loc=mu)
                        if theta>=0 and theta<=np.pi/2:
                            looking=False
                    weights=[np.sin(theta),np.cos(theta)]
                return phases,weights
            else:
                raise NoUException()

        elif method=='relative error':
            for w in weights:
                e=np.random.normal(loc=0,scale=kwargs['sigma'])
                r_w=w+e*w
                if r_w <= 0:
                    logging.debug('Warning, simulated weight < 0, setting to 0')
                    r_w=0
                r_weights.append(r_w)
        elif method=='exact':
            if u in phases:
                if len(phases)==1:
                    logging.debug('Convex hull was goal only')
                    return 'Success'
                i=phases.index(u)
                phases.remove(u)
                weights=np.delete(weights,i)
                weights=weights/sum(weights)
                return phases,weights
            else:
                logging.debug('Warning sampled point did not contain u')
                return(None,None)
        else:
            print('Error, unknown relative weight method')
            raise Exception()

        return phases,r_weights

    def get_mass_weights(self,p):
        phases,weights=self.get_molar_weights(p)
        wc=wt_converter()
        mass_weights=[
            wc.get_molar_mass(ph)[0]*w for ph,w in zip(phases,weights)]
        return (phases,mass_weights)

    def get_molar_weights(self,p):
        found=False
        weights=None
        phases=None
        for i,v in self.pure_phases.items():
            if np.allclose(p,v,atol=1e-3):
                phases=[i]
                weights=[1]
                return(phases,weights)
        for n,j in enumerate(self.triangles):
            Z=np.array([self.pure_phases[i] for i in j])
            if self.in_hull(Z,p):
                weights=self.get_bary_decomp(Z,p)
                phases=copy.deepcopy(j)
                if found==True:
                    for i in self.pure_phases.values():
                        print(i)
                    raise Exception('Error, found in >1 triangle')
                found=True
        if not found:
            raise Exception('Error, not in any triangle specific')
        '''
        fig,ax=plt.subplots(1)
        self.show(ax)
        phase_c=np.array([self.pure_phases[x] for x in phases])
        ax.scatter(phase_c[:,0],phase_c[:,1],marker='x')
        ax.scatter([p[0]],[p[1]],marker='o')
        plt.show()
        '''
        return(phases,weights)

    def get_triangle_index(self,p):
        print('Getting triangle index')
        found_triangles=[]
        for n,j in enumerate(self.triangles):
            print(n,j)
            Z=np.array([self.pure_phases[i] for i in j])
            if self.in_hull(Z,p):
                print('Contains')
                found_triangles.append(n)
        if len(found_triangles)==1:
            print('Found only one triangle :)')
            return found_triangles[0]
        elif len(found_triangles>1):
            print('Error found more than one triangle')
        else:
            print('Error, did not find containing triangle')
        return False

    def filter_omega(self,omega,void_triangles):
        logging.debug(f'orginal omega length {len(omega)}')
        new_omega=[]
        Zs=[]
        for i in void_triangles:
            Zs.append(
                np.array([self.pure_phases[j] for j in self.triangles[i]]))
        for p in omega:
            stay=True
            for Z in Zs:
                if self.in_hull(Z,p):
                    stay=False
            if stay:
                new_omega.append(p)
        omega=np.array(new_omega)
        logging.debug(f'final omega length {len(omega)}')
        return omega

    def get_highest_purity(self,points,u):
        highest_purity=0
        for p in points:
            ps,ws = self.get_mass_weights(p)
            purity=ws[ps.index(u)]
            if purity>highest_purity:
                highest_purity=purity
        return highest_purity

    def in_hull(self,points, x,):
        hull=Delaunay(points)
        return hull.find_simplex(x)>=0

    def get_bary_decomp(self,points,x):
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="A_eq does not appear to be of full row rank. To improve performance, check the problem formulation for redundant equality constraints.")
            lp = linprog(c, A_eq=A, b_eq=b)
        return (lp.x)

    def random_start(self,phase):
        corners=[]
        for i in self.triangles:
            if phase in i:
                for j in i:
                    if j not in corners:
                        if j != phase:
                            corners.append(j)

    def show(self,ax,omega):
        for t in self.triangles:
            triangle=[self.pure_phases[x] for x in t]
            for n,i in enumerate(triangle):
                for j in triangle[n+1:]:
                    ax.plot([i[0],j[0]],[i[1],j[1]])
        return ax


class PhaseFractionSimulatorComp(PhaseFractionSimulator):

    def __init__(self,file,converter,phase_field,na=None):
        with open(file, 'rb') as f:
            in_dict=pickle.load(f)
        entries=[]
        for k,v in in_dict.items():
            entries.append(ComputedEntry(k,v))
        self.pd=PhaseDiagram(entries)
        self.get_comp=converter
        if phase_field != [x.symbol for x in self.pd.elements]:
            raise Exception(
                'Phase field from species dict does not match pymatgen')
        self.phase_field=phase_field
        self.na=na
        self.max_purity=0

    def get_molar_weights(self,p):
        comp=self.get_comp(p)
        ps=[]
        ws=[]
        for k,v in self.pd.get_decomposition(Composition(comp)).items():
            if v < 1e-5:
                pass
            else:
                ps_d=k.composition.as_dict()
                phase=""
                for a,b in ps_d.items():
                    phase+=a+" "
                    if b.is_integer:
                        phase+=str(int(b))+" "
                    else:
                        phase+=str(b)+" "
                phase=phase[:-1]
                ps.append(phase)
                ws.append(v)
                phase_s=[k.composition.get_atomic_fraction(Element(x)) for x in
                         self.phase_field]
                if self.na is not None:
                    if abs(np.dot(self.na,phase_s))>1e-5:
                        raise Exception('Decomp phase was not charge neutral')
        return ps,ws

    def get_triangle_index(self,p):
        comp=self.get_comp(p)
        ps=[]
        '''
        print('---------start---------')
        print(comp)
        print(self.pd.get_decomposition(Composition(comp)))
        print('----------break--------')
        '''
        for k in self.pd.get_decomposition(Composition(comp)).keys():
            #print(k)
            ps.append(k)
        return ps

    def filter_omega(self,omega,void_triangles):
        new_omega=[]
        for p in omega:
            ps=self.get_triangle_index(p)
            if ps not in void_triangles:
                new_omega.append(p)
        omega=np.array(new_omega)
        return omega

    def show(self,omega):
        ps=[]
        for p in omega:
            tps=self.get_triangle_index(p)
            if not tps in ps:
                ps.append(tps)
        points=[]
        for t in ps:
            triangle=[]
            for k in t:
                x=[k.composition.get_atomic_fraction(Element(x)) for x in
                         self.phase_field]
                triangle.append(np.array(x))
            points.append(np.array(triangle))
        return points




