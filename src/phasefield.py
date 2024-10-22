import numpy as np
import scipy.stats
import math
from exceptions import *
from scipy.optimize import linprog
from simulating_phase_fractions import *
from visualise_square import *
import random

class Phase_Field:
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


    def set_mechanistic_parameters(self,k):
        self.cube_size=k['cube_size']
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

    def recompute_values_full(self):
        new_values=np.empty((len(self.s),len(self.omega)))
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
            '''
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
            '''
            raise ZeroPException()
        if np.any(p<0):
            if np.any(p< -1*tol):
                raise Exception('a P was < 0')
            else:
                p[p<0]=0
        if np.isnan(p).any():
            raise Exception('P contains NAN')
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

