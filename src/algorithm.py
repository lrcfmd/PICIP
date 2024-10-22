import numpy as np
import scipy.stats
import math
from exceptions import *
from scipy.optimize import linprog
from simulating_phase_fractions import *
from visualise_square import *
from scipy.stats import beta
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import random
import cdd as pcdd

class all_information:
    #overall class that contains most functions for processing information
    #aims not to handle input/ouput
    #aims to not be specific to use cases
    #This is shared version. My private version has much more functionality,
    #please be in contact for more details

    def setup(
            self,charge_vector,cube_size=100,basis=None,contained_point=None,
            show=True):
        #charge vector: list(n), expected formal charges for each element.
        #contained point: list(n), fractional composition of any charge neutral 
        #point.
        #cube size: integer: controlls resolution of discretisation.
        #basis, np.array((n-2,n)): if false prints computed basis, 
        #else uses basis given to construct grid.

        #converts params to right format and calls grid creator
        normala=np.array(charge_vector)
        normalb=np.ones(normala.shape)
        normal_vectors=np.stack((normala,normalb),axis=0)
        if contained_point is None:
            contained_point=self.get_contained_point(normal_vectors)
        contained_point=(cube_size*np.array(contained_point)
                         /sum(contained_point))
        self.create_omega_constrained(
            normal_vectors,cube_size,contained_point,basis=basis,show=show)

    def setup_from_species_dict(
            self,species_dict,cube_size=30,basis=None,contained_point=None,
            show=True):
        #species_dict: key of element string and value of formal charge
        #contained_point: any point in phase field
        #cube_size: resolution of grid
        #basis:optiional basis to use
        charge_vector=[]
        phase_field=[]
        for key,value in species_dict.items():
            charge_vector.append(value)
            phase_field.append(key)
        self.setup(
            charge_vector,cube_size,basis,contained_point=contained_point,
            show=show)
        self.phase_field=phase_field

    def setup_b(
            self,phase_field,cube_size=100,create_heatmap=False,basis=None,
            contained_point=None,show=True):
        #dim: integer, number of elements
        #cube size: integer: controlls resolution of discretisation.
        #basis, np.array((n-1,n)): if false prints computed basis, 
        #else uses basis given to construct grid.

        #takes setup params and cosntructs finite grid spanning phase field.
        #uses charge neutrality and fractional representation to reduce
        #dimensionality of grid to num elements-1.
        self.phase_field=phase_field
        dim=len(phase_field)
        normal_vectors=np.array([[1]*len(phase_field)])
        if contained_point==None:
            contained_point=np.array([cube_size/dim]*dim)
        else:
            contained_point=(cube_size*np.array(contained_point)
                             /sum(contained_point))
        self.create_omega_constrained(
            normal_vectors,cube_size,contained_point,basis=basis,show=show)


    def set_mechanistic_parameters(self,k,use_beta=True):
        self.cube_size=k['cube_size']
        if use_beta:
            self.n_l=k['n_l']
            self.n_p=k['n_p']
        else:
            self.use_fib=k['use_fib']
            self.demi_len=k['demi_len']
            self.line_len=k['line_len']
            self.total_mass=k['total_mass']

    def cartesian(self,arrays, out=None):
        #function for building grid (cartesian product)
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        #m = n / arrays[0].size
        m = int(n / arrays[0].size) 
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
            #for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        return out

    def find_orthonormal(self,A):
        #finds a vector orthonormal to the column space of A
        rand_vec=np.random.rand(A.shape[0],1)
        A = np.hstack((A,rand_vec))
        b = np.zeros(A.shape[1])
        b[-1] = 1
        x = np.linalg.lstsq(A.T,b,rcond=None)[0]
        return x/np.linalg.norm(x)

    def create_omega_constrained(
            self,normal_vectors,cube_size,contained_point,basis=None,
            show=True):
        #takes setup params and contructs finite grid spanning phase field.
        #uses charge neutrality and fractional representation to reduce
        #dimensionality of grid to num elements-2.

        dim=normal_vectors.shape[1]
        plane_dim = dim-len(normal_vectors)

        if basis is None:
            x=np.empty((plane_dim,dim))
            A = normal_vectors.T
            for i in range(plane_dim):
                x[i] = self.find_orthonormal(A)
                A = np.hstack((A,np.array([x[i]]).T))
            self.basis=x
            if show:
                print('Basis used:')
                print("np."+repr(self.basis))
        else:
            self.basis=basis

        self.contained_point=contained_point

        #generate big grid of points (centred on contained p and bounded
        #by cube of length the diagonal of n-dim cube
        max_length = np.sqrt(dim*cube_size**2)
        max_co = math.floor(max_length)
        ii = np.array(range(-max_co,max_co+1,1),dtype='float64')
        if plane_dim == 3:
            omega=self.cartesian((ii,ii,ii))
        elif plane_dim ==2:
            omega=self.cartesian((ii,ii))
        elif plane_dim==1:
            omega=self.cartesian((ii))
        elif plane_dim==4:
            omega=self.cartesian((ii,ii,ii,ii))
        elif plane_dim==5:
            omega=self.cartesian((ii,ii,ii,ii,ii))
        elif plane_dim==6:
            omega=self.cartesian((ii,ii,ii,ii,ii,ii))
        else:
            raise Exception(
                'Error, plane dim: '+str(plane_dim)+', is not implemented')

        #get constraint equations specifying (n-1) dim hyperplanes
        #coresponding to 0 of each element
        line_coefficients=self.get_constraint_lines()
        #delete points the dont satisfy equations

        for line in line_coefficients:
            omega=np.delete(
                omega,np.where(
                    np.einsum(
                        '...i,i->...',omega,line[:-1])<-1*line[-1]),
                axis=0)

        self.omega=omega #the grid of points
        self.constrained_dim=plane_dim #dimension of constrained space
        self.normal_vectors=normal_vectors #normal vectors
        self.cube_size=cube_size #resolution of discretisation

    def convert_to_standard_basis(self,omega,norm=1):
        #converts list of points in constrained basis to fractional
        #representation
        A=self.basis
        p_standard=self.contained_point+np.einsum('ji,...j->...i',A,omega)
        if p_standard.ndim==1:
            p_standard=norm*p_standard/np.sum(p_standard)
        else:
            norma=np.sum(p_standard,axis=1)
            p_standard=(p_standard.T/norma).T
        return p_standard

    def convert_point_to_constrained(self,point):
        #converts list of points in fractional representation to constrained
        #representation
        point=np.array(point)
        if point.ndim==1:
            point=np.array(point)*self.cube_size/sum(point)
            point_s=point-self.contained_point
            point=np.einsum('ij,j',self.basis,point_s)
            return point
        else:
            norma=np.sum(point,axis=1)/self.cube_size
            point=(point.T/norma).T
            point=point-self.contained_point
            point=np.einsum('ij,...j->...i',self.basis,point)
            return point

    def get_constraint_lines(self):
        #function that uses the assigned basis vectors to find the planes
        #corresponding to x_i>0 in constrained space
        line_coefficients=[]
        for i in range(self.basis.shape[1]):
            coefficients=[]
            for j in range(self.basis.shape[0]):
                coefficients.append(self.basis[j][i])
            coefficients.append(1*self.contained_point[i])
            line_coefficients.append(np.array(coefficients))
        return line_coefficients

    def convert_sigma_to_constrained(self,sigma):
        sigma=np.einsum('ij,jk,kl',self.basis,sigma,self.basis.T)
        return sigma

    def compute_k_sig(self,moles,formulas,moles_error):
        #function to return the mean and sigma matrix for a sample
        #moles should be relative number of moles for each phase
        #formulas give standard representation for each phase
        #moles error gives estimated error for each phase
        moles=np.array(moles)
        moles_error=np.array(moles_error)
        formulas=np.array(formulas)
        formulas_projected=np.array([np.array(x)/sum(x) for x in formulas])
        if len(formulas)!=len(moles) or len(moles)!=len(moles_error):
            raise Exception(
                'Error, moles,formulas,moles_error of different length')
        #compute average composition in standard rep and associated variance
        merged_mean=np.einsum('i,ij->j',moles,formulas_projected)
        merged_mean_var=np.einsum('i,ij->j',moles_error**2,formulas_projected**2)
        #normalise
        merged_mean_std=merged_mean_var**0.5*self.cube_size/np.sum(merged_mean)
        merged_mean*=self.cube_size/np.sum(merged_mean)
        #construct sigma
        merged_sigma=np.diag(merged_mean_std**2)
        #convert to constrained
        constrained_mean=self.convert_point_to_constrained(merged_mean)
        constrained_sigma=self.convert_sigma_to_constrained(merged_sigma)
        return (constrained_mean,constrained_sigma)

    def add_simulator_e(self,pure_phases,triangles):
        for key,value in pure_phases.items():
            pure_phases[key]=self.convert_point_to_constrained(value)
        self.simulator=PhaseFractionSimulator(pure_phases,triangles)

    def add_simulator_c(self):
        file="".join(self.phase_field)+".txt"
        if self.normal_vectors.shape[0]==2:
            na=self.normal_vectors[0]
        else:
            na=None
        self.simulator=PhaseFractionSimulatorComp(
            file,self.get_composition_string,self.phase_field,na)

    def get_errored_mass_weights(self,point,method,u,**kwargs):
        return self.simulator.get_errored_mass_weights(point,method,u,**kwargs)

    def get_errored_molar_weights(self,point,method,u,**kwargs):
        return self.simulator.get_errored_molar_weights(point,method,u,**kwargs)

    def get_composition_string(self,p):
        #function to return list of composition string for each constrained
        #point in array pos
        comps=[]
        p_s=self.convert_to_standard_basis(p)
        p_s=p_s/sum(p_s)
        comp=""
        for x,l in zip(p_s,self.phase_field):
            comp+=l+str(x)+" "
        comps.append(comp)
        if len(comps)==1:
            return comps[0]
        else:
            return comps

    def random_start_with_u(self,min_u,size=1,exclude_u=False):
        #picks random points from omega untill one is found with u as a vertex
        look=True
        i=0
        ps=[]
        look=0
        while(look<size):
            i+=1
            if i > 99999:
                raise Exception('Suitable start not found after 99999 attempts')
            p=self.random_point_constrained()
            result=self.simulator.get_mass_weights(p)
            phases=result[0]
            try:
                n=phases.index(self.unknown_phase)
            except:
                pass
            else:
                if not list(p) in ps:
                    ws=np.array(result[1])/sum(result[1])
                    if exclude_u:
                        if ws[n]<0.99:
                            if ws[n]>min_u:
                                ps.append(list(p))
                                look+=1
                    else:
                        if ws[n]>min_u:
                            ps.append(list(p))
                            look+=1
        return np.array(ps)

    def random_point_constrained(self,n=1):
        #function to return a random point or list of points from omega
        choice = random.randrange(len(self.omega))
        indicis = range(len(self.omega))
        r_indicis = np.random.choice(indicis,n,replace=False)
        ps=self.omega[r_indicis]
        if n==1:
            return ps[0]
        else:
            return ps

    def add_sksig(self,s,k,sig,make_p=False,exclude_nan=True):
        if make_p:
            p=self.create_density_from_sample(
                s,k,sig,n=self.demi_len,
                points_per_line=self.line_len,fib=self.use_fib)
            if exclude_nan:
                if np.isnan(p).any():
                    return
            if hasattr(self,'values_full'):
                self.values_full=np.append(self.values_full,[p],axis=0)
            else:
                self.values_full=p[np.newaxis,:]
            if hasattr(self,'values'):
                self.values*=p
            else:
                self.values=p
        if hasattr(self,'s'):
            self.s=np.append(self.s,[s],axis=0)
        else:
            self.s=s[np.newaxis,:]
        if hasattr(self,'k'):
            self.k=np.append(self.k,[k],axis=0)
        else:
            self.k=k[np.newaxis,:]
        if hasattr(self,'sig'):
            self.sig=np.append(self.sig,[sig],axis=0)
        else:
            self.sig=sig[np.newaxis,:]

    def add_knowns_weights(self,ps,ws):
        if hasattr(self,'knowns_weights'):
            self.knowns_weights.append((ps,ws))
        else:
            self.knowns_weights=[(ps,ws)]

    def get_score(self,method):
        if method=='d_g_mu':
            mean=self.get_mean(f=self.values)
            return np.linalg.norm(mean-self.goal)
        elif method=='true_second_moment':
            dgC=np.linalg.norm(self.omega-self.goal,axis=1)
            return np.sum(dgC*self.values)
        elif method=='closest distance':
            chosen_point=self.get_closest_point(self.s,index=False)
            return np.linalg.norm(self.goal-chosen_point)
        elif method=='purity':
            return self.simulator.max_purity
        else:
            raise Exception('Error: unknown scoring method')

    def get_mean(self,f=None,omega=None):
        if omega is None:
            omega=self.omega
        if f is None:
            f=self.values
        f=f/np.sum(f)
        u=np.swapaxes(np.broadcast_to(f,(omega.shape[1],len(f))),0,1)*omega
        mean=np.sum(u,axis=0)
        return mean

    def get_closest_point(self,next_points,index=True):
            mindist=9999
            closest_point_index=None
            closest_point=None
            goal=self.convert_to_standard_basis(self.goal)/self.cube_size
            for n,point in enumerate(next_points):
                spoint=self.convert_to_standard_basis(point)/self.cube_size
                dist=np.linalg.norm(spoint-goal)
                if dist<mindist:
                    mindist=dist
                    closest_point_index=n
                    closest_point=point
            if index:
                return closest_point_index
            else:
                return closest_point

    def recompute_values_full(self,beta_method=True):
        new_values=np.empty((len(self.s),len(self.omega)))
        if beta_method:
            del self.values_full
            del self.values
            for s,ws,ks,std in zip(self.s,self.weights,self.knowns,self.beta_std):
                self.compute_p_beta(ws,ks,s,std)
        else:
            for n in range(len(self.s)):
                new_values[n]=self.create_density_from_sample(
                    *self.get_sksig(n),n=self.demi_len,
                    points_per_line=self.line_len,fib=self.use_fib)
            self.values_full=new_values

    def create_density_from_sample(
            self,sample_point,avg_known,sigma,n=800, normalize=True,
            points_per_line=500,fib=False,tol=1e-5):

        uniform_demisphere=None
        if fib:
            uniform_demisphere=self.get_uniform_demisphere_fib(
                sample_point,avg_known,n=n)
        else:
            uniform_demisphere=self.get_uniform_demisphere(
                sample_point,avg_known,n=n)
        z=[]
        points=[]
        lines=self.get_constraint_lines()

        for p in uniform_demisphere:
            v=p-sample_point

            ts=[]
            for k in lines:
                ts.append(self.get_t(k,p,v))
            mint=99999
            for j in ts:
                if j > 0:
                    if j < mint:
                        mint=j
            if mint==99999:
                raise Exception('Point of intersection not found')
            intersect=p+mint*v
            l=np.linalg.norm(sample_point-intersect)

            f=None
            if self.constrained_dim == 2:
                A=self.get_basis(v).T[1]
                sigma_perp=np.einsum('j,jk,k',A.T,sigma,A)
                sigma_perp=np.sqrt(sigma_perp)
                y=np.einsum('j,j',A,sample_point-avg_known)
                f=scipy.stats.norm.pdf(y,0,sigma_perp)
            else:
                A=self.get_basis(v).T[1:]
                sigma_perp=np.einsum('ij,jk,kl',A,sigma,A.T)
                y=np.einsum('ij,j',A,sample_point-avg_known)
                f=scipy.stats.multivariate_normal.pdf(
                    y,np.zeros(self.constrained_dim-1),sigma_perp)
            values=[f/l]*points_per_line
            z.extend(values)
            l_ps=[(sample_point+x/(points_per_line-1)*(intersect-sample_point))
                  for x in range(points_per_line)]
            points.extend(l_ps)

        z=np.array(z)
        points=np.array(points)
        p=scipy.interpolate.griddata(
            points,z,self.omega,method='linear',fill_value=0)

        if sum(p)==0:
            raise ZeroPException()
        if np.any(p<0):
            if np.any(p< -1*tol):
                raise Exception('a P was < 0')
            else:
                p[p<0]=0
        if np.isnan(p).any():
            raise Exception('P contains NAN')

            if hasattr(self,'values_full'):
                self.values_full=np.append(self.values_full,[p],axis=0)
            else:
                self.values_full=p[np.newaxis,:]
            if hasattr(self,'values'):
                self.values*=p
            else:
                self.values=p
        if hasattr(self,'s'):
            self.s=np.append(self.s,[s],axis=0)
        else:
            self.s=s[np.newaxis,:]
        if hasattr(self,'k'):
            self.k=np.append(self.k,[k],axis=0)
        else:
            self.k=k[np.newaxis,:]
        if hasattr(self,'sig'):
            self.sig=np.append(self.sig,[sig],axis=0)
        else:
            self.sig=sig[np.newaxis,:]
        return(p/np.sum(p))

    def show(self,triangles=True):
        if triangles:
            fig,axs=plt.subplots(1,3)
            axs[0].scatter(self.omega[:,0],self.omega[:,1],c=p)
            axs[0].scatter([sample_point[0]],[sample_point[1]],marker='x',label='s')
            axs[0].scatter([avg_known[0]],[avg_known[1]],marker='x',label='k')
            cp=self.convert_point_to_constrained(self.contained_point)
            axs[0].scatter([cp[0]],[cp[1]],marker='x',label='g')
            axs[1].scatter(points[:,0],points[:,1],c=z)
            points=self.simulator.show(self.omega)
            if points[0].shape[1]==4:
                tpoints=[]
                for t in points:
                    t=self.convert_point_to_constrained(t)
                    tpoints.append(t)
                points=tpoints
            for t in points:
                for n,i in enumerate(t):
                    for j in t[n+1:]:
                        axs[2].plot([i[0],j[0]],[i[1],j[1]])
            plt.legend()
            plt.show()

    def get_uniform_demisphere_fib(self,sample_point,avg_known,n=500):
        if self.constrained_dim==3:
            goldenRatio=(1+5**0.5)/2
            i=np.arange(0,n)
            theta=2*np.pi*i/goldenRatio
            phi=np.arccos(1-2*(i+0.5)/n)
            points=np.array(
                [np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),
                 np.cos(phi)])
            points=points.T
        elif self.constrained_dim==2:
            theta=np.linspace(0,2*np.pi,n)
            points=np.array([np.cos(theta),np.sin(theta)])
            points=points.T
            #print(points.shape)
            
        else:
            print('Warning, golden angle method not implemented for' +
                  str(self.constrained_dim +2) + "elements, using regular" +
                  " method")
            return self.get_uniform_demisphere(sample_point,avg_known,n)
        points=points/99+sample_point
        a=points-sample_point
        b=sample_point-avg_known
        c=np.dot(a,b)
        points=np.delete(points,np.where(np.dot(a,b)<=0),axis=0)
        return points

    def get_uniform_demisphere(self,sample_point,avg_known,n=500):
        points=np.random.multivariate_normal(
               np.zeros(len(sample_point)),np.eye(len(avg_known)),size=n)
        points=normalize(points)
        points=points+sample_point
        a=points-sample_point
        b=sample_point-avg_known
        c=np.dot(a,b)
        points=np.delete(points,np.where(np.dot(a,b)<=0),axis=0)
        #fig=plt.figure()
        #ax=fig.add_subplot(projection='3d')
        #ax.scatter(points[:,0],points[:,1],points[:,2],s=1)
        #plt.show()
        return points

    def get_t(self,line,a,b):
        t_co=0
        c_co=0
        for n,i in enumerate(line[:-1]):
            t_co+=i*b[n]
            c_co+=i*a[n]
        c_co+=line[-1]
        t=-c_co/t_co
        return t

    def get_basis(self, estimated_direction):
        dim = len(estimated_direction)
        #function to caluclate basis for each sample
        #basis: numpy (dim,dim) array such such that
        # basis matmul (a point) gives the point in
        # representation where first element is size in estimated_direction and
        # subsequent elements are sizes in orthogonal directions
        A = np.transpose(np.array([estimated_direction]))
        for i in range(dim-1):
            x = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x]).T))
        return A

    def normalise_values(self,thiccen_cones=False,increment=0.4):
        if np.sum(self.values)==0:
            if thiccen_cones:
                if self.predicted_error>2:
                    raise ConflictingDataException()
                #predicted error and recomputing all cones
                self.predicted_error+=increment
                logging.debug(
                    'Null density: setting predicted error to '
                    +f'{self.predicted_error}')
                wt_convert=wt_converter()
                for n,(ps,ws) in enumerate(self.knowns_weights):
                    predicted_errors=[self.predicted_error for x in ps]
                    #print("B",ps) TODO occasional error
                    moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
                        ps,ws,phase_field=self.phase_field,
                        weights_error=predicted_errors,
                        total_mass=self.total_mass)
                    k,sig=self.compute_k_sig(moles,formulas_standard,moles_error)
                    self.sig[n]=sig
                    p=self.create_density_from_sample(
                        *self.get_sksig(n),n=self.demi_len,
                        points_per_line=self.line_len,fib=self.use_fib)
                    self.values_full[n]=p
                self.values=np.prod(self.values_full,axis=0)
                thic_enough=self.normalise_values(thiccen_cones=True)
            else:
                raise ConflictingDataException()
        self.values=self.values/np.sum(self.values)

    def get_contained_point(self,A):
        c=np.zeros(A.shape[1])
        c[0]=1
        b=[0]*A.shape[0]
        b[-1]=1
        return linprog(c,A_eq=A,b_eq=b).x

    def get_sksig(self,i):
        return (self.s[i],self.k[i],self.sig[i])

    def set_next_batch(self,method,batch_size,p=None,use_mean=True):
        next_points=np.empty((batch_size,self.constrained_dim))
        if method=='lastline':
            point=self.points[-1]
            endpoint=self.get_end_points('constrained')[-1]
            v=endpoint-point
            distance=np.linalg.norm(v)
            for i in range(1,batch_size):
                next_points[i-1]=point+i/(batch_size)*v
            next_points[batch_size-1]=endpoint
        if method=='random':
            next_points=self.omega[np.random.choice(
                range(len(self.omega)),size=batch_size,replace=False)]
        if method=='sample from p':
            if use_mean:
                if not self.voided_triangles:
                    if np.count_nonzero(p)>=batch_size:
                        next_points=self.omega[np.random.choice(
                            range(len(self.omega)),size=batch_size-1,
                            replace=False,p=p)
                        ]
                    else:
                        next_points=self.omega[p.nonzero()]
                    next_points=np.append(
                        next_points,[self.get_mean(f=p)],axis=0)
                else:
                    mean=self.get_mean(f=p)
                    ps=self.simulator.get_triangle_index(mean)
                    if not ps in self.voided_triangles:
                        if np.count_nonzero(p)>=batch_size-1:
                            next_points=self.omega[np.random.choice(
                                range(len(self.omega)),size=batch_size-1,
                                replace=False,p=p)
                            ]
                        else:
                            next_points=self.omega[p.nonzero()]
                        next_points=np.append(next_points,[self.get_mean(f=p)],
                                              axis=0)
                    else:
                        if np.count_nonzero(p)>=batch_size:
                            next_points=self.omega[np.random.choice(
                                range(len(self.omega)),size=batch_size,
                                replace=False,p=p)
                            ]
                        else:
                            next_points=self.omega[p.nonzero()]
            else:
                if np.count_nonzero(p)>=batch_size:
                    next_points=self.omega[np.random.choice(
                        range(len(self.omega)),size=batch_size,replace=False,
                        p=p)
                    ]
                else:
                    next_points=self.omega[p.nonzero()]
        self.next_batch=next_points

    def set_end_points_from_s(self):
        self.end_points=[]
        for s,k in zip(self.s,self.k):
            point=self.convert_to_standard_basis(s)
            line=point-self.convert_to_standard_basis(k)
            min_steps=9999
            for pi,vi in zip(point,line):
                if vi < 0:
                    if pi/(-1*vi) < min_steps:
                        min_steps = -1*pi/vi
            final_point = np.add(point,line*min_steps)
            self.end_points.append(
                self.convert_point_to_constrained(final_point))

    def show_sim(self,method):
        plotter=Square(self)
        plotter.add_corners(
            [[0,2,0,3],[0,0,2,3],[2,0,0,1]],["Al2O3","B2O3","Li2O"])
        plotter.create_fig(corners=True)
        self.set_end_points_from_s()
        for n,(p,e) in enumerate(zip(self.s,self.end_points)):
            plotter.plot_point(p,c='red',label=str(n))
            plotter.plot_line(p,e)
        plotter.plot_point(self.goal,label='goal') 
        for j in self.simulator.triangles:
            plotter.plot_triangle_line([self.simulator.pure_phases[i] for i in j])
        plotter.show()

    def map_to_ab_line(self,point, A, B):
        """
        Maps a point to a parameter t in [0, 1] along the line from A to B.
        """
        AB = B - A  # Vector from A to B
        AP = point - A  # Vector from A to the point
        t = np.dot(AP, AB) / np.dot(AB, AB)  # Projection of AP onto AB, normalized by the length of AB
        return t

    def compute_p_beta(self,weights,k_c,sample,std,n_l=None,n_p=None):
            if n_l is None:
                n_l=self.n_l
            if n_p is None:
                n_p=self.n_p
            mean=weights[1]
            # Calculate a and b using the mean and sigma
            alpha=-1
            beta_param=-1
            while alpha<0 and beta_param < 0:
                variance = std ** 2
                common_factor = mean * (1 - mean) / variance - 1
                alpha = mean * common_factor
                beta_param = (1 - mean) * common_factor
                std=std/1.2
            
            # Discretize the line between points a and b
            line_points=self.discretise_known_simplex_2d(k_c[0],k_c[1],sample,n_l)[1:-1]

            # Normalize these points to the range [0, 1]
            normalised_points = [self.map_to_ab_line(point, k_c[0], k_c[1]) for point in line_points]

            # Get the probability density function (PDF) of the Beta distribution
            beta_pdf = beta.pdf(normalised_points, alpha, beta_param)

            '''
            scaled_pdf = beta_pdf / np.max(beta_pdf)  # Scale for visualization
            # Plot the results
            plt.figure(figsize=(8, 6))
            plt.plot(normalized_points, scaled_pdf, label="Beta PDF")
            plt.title(f"Beta Distribution with α={alpha}, β={beta_param} on the line from a to b")
            plt.xlabel("Normalized distance between a and b")
            plt.ylabel("Probability Density")
            plt.legend()
            plt.show()
            '''
            known_simplex=line_points
            if np.any(np.isinf(beta_pdf)):
                print('infa')
            if np.any(np.isnan(beta_pdf)):
                print('nana')
                print(alpha)
                print(beta_param)
                print(normalised_points)
                print(beta_pdf)

            known_pdf=beta_pdf/sum(beta_pdf)
            p=self.project_known_simplex(known_simplex,known_pdf,sample,n_p)
            if hasattr(self,'values_full'):
                self.values_full=np.append(self.values_full,[p],axis=0)
                self.values=np.prod(self.values_full,axis=0)
            else:
                self.values_full=p[np.newaxis,:]
                self.values=p

    def add_sample(self,knowns,weights,sample,std=0.02,n_l=None,n_p=None,
                   compute_p=True):

        weights=np.array(weights)/sum(weights)
        knowns=[Composition(k) for k in knowns]
        k_s=np.array([[k.get_atomic_fraction(el) for el in self.phase_field] for k in knowns])
        k_c=self.convert_point_to_constrained(k_s)

        if compute_p:
            self.compute_p_beta(weights,k_c,sample,std,n_l,n_p)
        if hasattr(self,'s'):
            self.s=np.append(self.s,[sample],axis=0)
        else:
            self.s=sample[np.newaxis,:]
        k=weights[0]*k_c[0]+weights[1]*k_c[1]
        if hasattr(self,'k'):
            self.k=np.append(self.k,[k],axis=0)
        else:
            self.k=k[np.newaxis,:]
        if hasattr(self,'weights'):
            self.weights=np.append(self.weights,[weights],axis=0)
        else:
            self.weights=weights[np.newaxis,:]
        if hasattr(self,'knowns'):
            self.knowns=np.append(self.knowns,[k_c],axis=0)
        else:
            self.knowns=k_c[np.newaxis,:]
        if hasattr(self,'beta_std'):
            self.beta_std.append(std)
        else:
            self.beta_std=[std]

    def project_known_simplex(self,known_simplex,known_pdf,sample,n_p,tol=1e-4):
        lines=np.array(self.get_constraint_lines())
        A=lines[:,:-1]
        b=-1*lines[:,-1]
        polygon_constraints={
                'A':A,
                'b':b
                }
        #polygon_constraints=
        points=[]
        values=[]
        for x,p in zip(known_simplex,known_pdf):
            line=[]
            direction=sample-x
            direction=direction/np.linalg.norm(direction)
            end_point=self.find_intersection(polygon_constraints,x,sample)
            l_ps=[(sample+x/(n_p-1)*(end_point-sample))
                  for x in range(n_p)]
            points+=l_ps
            values+=[p]*n_p

            #plotting
            '''
            plot=self.make_fig()
            plot.plot_point(sample)
            plot.plot_point(end_point)
            plot.show()
            '''

        #plotting
        '''
        plot=self.make_fig()
        plot.plot_p(points,values,s=2)
        plot.plot_points_np(known_simplex)
        plot.show()
        '''

        p=scipy.interpolate.griddata(
            points,values,self.omega,method='linear',fill_value=0)
        if sum(p)==0:
            raise ZeroPException()
        if np.any(p<0):
            if np.any(p< -1*tol):
                raise Exception('a P was < 0')
            else:
                p[p<0]=0
        if np.isnan(p).any():
            print('P contained nan')
            p=np.nan_to_num(p)
        return(p/np.sum(p))

    def line_through_point(self, x, sample, t):
        """/
        Parametric line equation: 
        Line starts at point x and passes through sample, parameterized by t.
        """
        return x + t * (sample - x)

    def find_intersection(self, polygon_constraints, x, sample, epsilon=1e-2):
        """
        Find the intersectio    n of the line starting at x and passing through sample
        with the edges of the polygon defined by polygon_constraints.
        
        polygon_constraints is an array of linear inequalities Ax <= b.
        """
        intersections = []
        
        # Parametric form of the line: x + t(sample - x), for t >= 0.
        direction = sample - x
        d=direction/np.linalg.norm(direction)

        #plotting
        '''
        plot=self.make_fig()
        plot.plot_point(x)
        plot.plot_point(sample)
        plot.plot_point(sample+d)
        plot.show()
        '''

        # Iterate through each edge of the polygon, given by the inequalities Ax <= b
        for i, (A_row, b_val) in enumerate(zip(polygon_constraints['A'], polygon_constraints['b'])):
            # Find t such that A @ (x + t * (sample - x)) = b (i.e., intersection)
            # This simplifies to A @ direction * t = b - A @ x
            A_dot_dir = np.dot(A_row, direction)
            
            if A_dot_dir != 0:  # Only if the direction and edge are not parallel
                t_intersect = (b_val - np.dot(A_row, x)) / A_dot_dir
                
                if t_intersect > 0:  # We only care about positive t values (forward in direction)
                    # Compute intersection point
                    intersection_point = self.line_through_point(x, sample, t_intersect)
                    if np.linalg.norm(intersection_point - x) > epsilon:
                        intersections.append(intersection_point)
        
        # Return the intersection point closest to x
        if len(intersections) > 0:
            distances = [np.linalg.norm(intersection - x) for intersection in intersections]
            closest_intersection = intersections[np.argmin(distances)]
            return closest_intersection
        else:
            print('no intersection found')
            return None  # No intersection found (this should not happen in a bounded polygon)
        
    def find_corners_edges(self,A,add_corners_to_knowns=False):
        '''
        gets corners and edges of polyhedra
        A:2d np matrix with columns as the constraining linear equations
        returns in standard rep
        '''
        A = A.T
        cp=self.contained_point
        dim=A.shape[0]
        plane_dim=dim-A.shape[1]
        x=np.empty((plane_dim,dim))
        for i in range(plane_dim):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        charge_basis=x
        cons=[]
        for i in range(dim):
            con=[0]*dim
            con[i]=1
            cons.append(np.array(con))
        cons_c=[]
        if plane_dim==2:
            for i in cons:
                a=np.dot(i,charge_basis[0])
                b=np.dot(i,charge_basis[1])
                c=np.dot(i,cp)
                cons_c.append([c,a,b])
        elif plane_dim==3:
            for i in cons:
                a=np.dot(i,charge_basis[0])
                b=np.dot(i,charge_basis[1])
                c=np.dot(i,charge_basis[2])
                d=np.dot(i,cp)
                cons_c.append([d,a,b,c])
        else:
            raise Exception('Not implemented for polytope above 3d')
        #create matric for polyhedra construction
        mat=pcdd.Matrix(cons_c,linear=False)
        mat.rep_type = pcdd.RepType.INEQUALITY
        #create polyhedron
        poly = pcdd.Polyhedron(mat)
        #get corners
        ext = poly.get_generators()
        corners=[]
        for i in ext:
            corners.append(i[1:])
        corners=np.array(corners)
        #get edges
        edges=[]
        adjacencies = list(poly.get_adjacency())
        for n,i in enumerate(adjacencies):
            for j in list(i):
                if j>n:
                    edges.append([n,j])
        corner_compositions=self.get_composition_string(corners)
        corners=cp+np.einsum('ji,...j->...i',charge_basis,corners)
        self.corners=corners
        self.edges=edges
        self.corner_compositions=corner_compositions
        if add_corners_to_knowns:
            knowns_mat=self.convert_standard_to_pymatgen(corners)
            self.add_knowns(knowns_mat)
             
    def make_fig(self):
        plot=Square(self)
        plot.make_binaries_as_corners()
        plot.create_fig()
        return plot

    def line_intersection(self,p1, p2, q1, q2):
        """
        Finds the intersection of two lines (p1->p2 and q1->q2).
        Returns the intersection point.
        """
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = q2[1] - q1[1]
        B2 = q1[0] - q2[0]
        C2 = A2 * q1[0] + B2 * q1[1]

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # The lines are parallel
        else:
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return np.array([x, y])

    def rotate_vector(self,v, angle,sample):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
        return np.dot(rotation_matrix, v)+sample

    def discretise_known_simplex_2d(self,start,stop,sample,n_l):
        ''' Discretises known simplex in 2d. 
theta=np.linspace(0,2*np.pi,n)
            points=np.array([np.cos(theta),np.sin(theta)])
            points=points.T
                '''
        # 1) Calculate vectors from the sample to start and stop
        v1 = start - sample
        v2 = stop - sample

        # 2) Compute the angle between the two vectors
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
        delta_angle = angle / (n_l - 1)  # Dividing the angle into n_l parts
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product<0:
            delta_angle=-1*delta_angle


        # 5) Generate points by rotating v1 by increments of delta_angle
        rotated_points = [start]  # Start with the initial point
        linea=stop-start
        for i in range(1, n_l):
            # Rotate the v1 vector by i * delta_angle
            rotated_vector = self.rotate_vector(v1, i * delta_angle,sample)
            intersection=self.line_intersection(sample,rotated_vector,start,stop)
            if intersection is not None:
                rotated_points.append(intersection)
            else:
                raise Exception('error finding line intersection')

        rotated_points = np.array(rotated_points)

        # 6) Visualize the result
        

        '''
        plt.figure(figsize=(6, 6))
        plt.plot([sample[0], start[0]], [sample[1], start[1]], 'b-')
        plt.plot([sample[0], stop[0]], [sample[1], stop[1]], 'b-')
        plt.scatter([start[0],stop[0]],[start[1],stop[1]])

        # Plot the start, stop, and sample points
        #plt.scatter(*start, color='green', label='Start')
        #plt.scatter(*stop, color='blue', label='Stop')
        #plt.scatter(*sample, color='red', label='Sample')

        # Plot lines from sample to each rotated point
        for point in rotated_points:
            print(point)
            #plt.plot(point)
            #plt.plot([sample[0], point[0]], [sample[1], point[1]], 'b-')
            #plt.scatter(*point, color='orange')  # Mark the rotated point
        plt.scatter(rotated_points[:,0],rotated_points[:,1])
        # Adjust the plot
        #plt.xlim([min([start[0], stop[0], sample[0]]) - 1, max([start[0], stop[0], sample[0]]) + 1])
        #plt.ylim([min([start[1], stop[1], sample[1]]) - 1, max([start[1], stop[1], sample[1]]) + 1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.legend()
        plt.title(f"Lines Emanating from Sample Point with Equal Delta Angle ({np.degrees(delta_angle):.2f}°)")
        plt.show()
        '''
        return rotated_points



                
                

