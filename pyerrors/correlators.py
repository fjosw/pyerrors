import autograd.numpy as np
from .pyerrors import *
from .fits import standard_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

class Corr:
    """The class for a correlator (time dependent sequence of pe.Obs).

    Everything, this class does, can be achieved using lists or arrays of Obs. 
    But it is simply more convenient to have a dedicated object for correlators. 
    One often wants to add or multiply correlators of the same length at every timeslice and it is inconvinient 
    to iterate over all timeslices for every operation. This is especially true, when dealing with smearing matrices.

    The correlator can have two types of content: An Obs at every timeslice OR a GEVP
    smearing matrix at every timeslice. Other dependency (eg. spacial) are not supported.

    """

    def __init__(self, data_input,padding_front=0,padding_back=0):
        #All data_input should be a list of things at different timeslices. This needs to be verified

        if not isinstance(data_input,list):
            raise TypeError('Corr__init__ expects a list of timeslices.')
        # data_input can have multiple shapes. The simplest one is a list of Obs.
        #We check, if this is the case
        if all([isinstance(item,Obs) for item in data_input]):
            self.content=[np.asarray([item]) for item in data_input]
            #Wrapping the Obs in an array ensures that the data structure is consistent with smearing matrices.
            self.N = 1 # number of smearings

        #data_input in the form [np.array(Obs,NxN)]
        elif all([isinstance(item,np.ndarray) or item==None for item in data_input]) and any([isinstance(item,np.ndarray)for item in data_input]):
            self.content = data_input

            noNull=[a for a in self.content if not (a is None)] #To check if the matrices are correct for all undefined elements
            self.N = noNull[0].shape[0]
            # The checks are now identical to the case above
            if self.N > 1 and noNull[0].shape[0] != noNull[0].shape[1]:
                raise Exception("Smearing matrices are not NxN")
            if  (not all([item.shape == noNull[0].shape for item in noNull])):
                raise Exception("Items in data_input are not of identical shape." + str(noNull))

        else: # In case its a list of something else.
            raise Exception ("data_input contains item of wrong type")

        #We now apply some padding to our list. In case that our list represents a correlator of length T but is not defined at every value.
        #An undefined timeslice is represented by the None object
        self.content = [None] * padding_front + self.content + [None] * padding_back
        self.T = len(self.content) #for convenience: will be used a lot


        self.gamma_method()


    def gamma_method(self):
        for item in self.content:
            if not(item is None):
                if self.N == 1:
                    item[0].gamma_method()
                else:
                    for i in range(self.N):
                        for j in range(self.N):
                            item[i,j].gamma_method()

    #We need to project the Correlator with a Vector to get a single value at each timeslice.
    #The method can use one or two vectors.
    #If two are specified it returns v1@G@v2 (the order might be very important.)
    #By default it will return the lowest source, which usually means unsmeared-unsmeared (0,0), but it does not have to
    def projected(self,vector_l=None,vector_r=None):
        if self.N == 1:
            raise Exception("Trying to project a Corr, that already has N=1.")
            #This Exception is in no way necessary. One could just return self
            #But there is no scenario, where a user would want that to happen and the error message might be more informative.

        self.gamma_method()

        if vector_l is None:
            vector_l,vector_r=np.asarray([1.] + (self.N - 1) * [0.]),np.asarray([1.] + (self.N - 1) * [0.])
        elif(vector_r is None):
            vector_r=vector_l

        if not vector_l.shape == vector_r.shape == (self.N,):
            raise Exception("Vectors are of wrong shape!")

        #We always normalize before projecting! But we only raise a warning, when it is clear, they where not meant to be normalized.
        if (not(0.95 < vector_r@vector_r < 1.05)) or (not(0.95 < vector_l@vector_l < 1.05)):
            print("Vectors are normalized before projection!")

        vector_l,vector_r = vector_l / np.sqrt((vector_l@vector_l)), vector_r / np.sqrt(vector_r@vector_r)

        newcontent = [None if (item is None) else np.asarray([vector_l.T@item@vector_r]) for item in self.content]
        return Corr(newcontent)

    #For purposes of debugging and verification, one might want to see a single smearing level. smearing will return a Corr at the specified i,j. where both are integers 0<=i,j<N.
    def smearing(self, i, j):
        if self.N == 1:
            raise Exception("Trying to pick smearing from projected Corr")
        newcontent=[None if(item is None) else item[i, j] for item in self.content]
        return Corr(newcontent)


    #Obs and Matplotlib do not play nicely
    #We often want to retrieve x,y,y_err as lists to pass them to something like pyplot.errorbar
    def plottable(self):
        if self.N != 1:
            raise Exception("Can only make Corr[N=1] plottable") #We could also autoproject to the groundstate or expect vectors, but this is supposed to be a super simple function.
        x_list = [x for x in range(self.T) if (not self.content[x] is None)]
        y_list = [y[0].value for y in self.content if (not y is None)]
        y_err_list = [y[0].dvalue for y in self.content if (not y is None)]

        return x_list, y_list, y_err_list

    #symmetric returns a Corr, that has been symmetrized.
    #A symmetry checker is still to be implemented
    #The method will not delete any redundant timeslices (Bad for memory, Great for convenience)
    def symmetric(self):

        if self.T%2 != 0:
            raise Exception("Can not symmetrize odd T")

        newcontent = [self.content[0]]
        for t in range(1, self.T):
            if (self.content[t] is None) or (self.content[self.T - t] is None):
                newcontent.append(None)
            else:
                newcontent.append(0.5 * (self.content[t] + self.content[self.T - t]))
        if(all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent)

    def anti_symmetric(self):

        if self.T%2 != 0:
            raise Exception("Can not symmetrize odd T")

        newcontent=[self.content[0]]
        for t in range(1, self.T):
            if (self.content[t] is None) or (self.content[self.T - t] is None):
                newcontent.append(None)
            else:
                newcontent.append(0.5 * (self.content[t] - self.content[self.T - t]))
        if(all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent)



    #This method will symmetrice the matrices and therefore make them positive definit.
    def smearing_symmetric(self):
        if self.N > 1:
            transposed = [None if (G is None) else G.T for G in self.content]
            return 0.5 * (Corr(transposed)+self)
        if self.N == 1:
            raise Exception("Trying to symmetrize a smearing matrix, that already has N=1.")


    #We also include a simple GEVP method based on Scipy.linalg
    def GEVP(self, t0, ts):
        if (self.content[t0] is None) or (self.content[ts] is None):
            raise Exception("Corr not defined at t0/ts")
        G0, Gt = np.empty([self.N, self.N], dtype="double"), np.empty([self.N, self.N], dtype="double")
        for i in range(self.N):
            for j in range(self.N):
                G0[i, j] = self.content[t0][i, j].value
                Gt[i, j] = self.content[ts][i, j].value

        sp_val,sp_vec = scipy.linalg.eig(Gt, G0)
        sp_vec = sp_vec[:,np.argmax(sp_val)] #we only want the eigenvector belonging to the biggest eigenvalue.
        sp_vec = sp_vec/np.sqrt(sp_vec@sp_vec)
        return sp_vec

    def deriv(self, symmetric=False): #Defaults to forward derivative f'(t)=f(t+1)-f(t) 
        if not symmetric:
            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t+1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t + 1] - self.content[t])
            if(all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding_back=1)
        if symmetric:
            newcontent = []
            for t in range(1, self.T-1):
                if (self.content[t-1] is None) or (self.content[t+1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(0.5 * (self.content[t + 1] - self.content[t - 1]))
            if(all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding_back=1, padding_front=1)

    #effective mass at every timeslice
    def m_eff(self, periodic=False):
        if self.N != 1:
            raise Exception("Correlator must be projected before getting m_eff")
        if not periodic:
            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / self.content[t + 1])
            if(all([x is None for x in newcontent])):
                raise Exception("m_eff is undefined at all timeslices")

            return np.log(Corr(newcontent, padding_back=1))

        else: #This is usually not very stable. One could default back to periodic=False.
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None)or (self.content[t - 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append((self.content[t + 1] + self.content[t - 1]) / (2 * self.content[t]))
            if(all([x is None for x in newcontent])):
                raise Exception("m_eff is undefined at all timeslices")
            return np.arccosh(Corr(newcontent,padding_back=1, padding_front=1))


    #We want to apply a pe.standard_fit directly to the Corr using an arbitrary function and range.
    def fit(self, function, fitrange=None):
        if self.N != 1:
            raise Exception("Correlator must be projected before fitting")

        if fitrange is None:
            fitrange=[0, self.T]

        xs = [x for x in range(fitrange[0], fitrange[1]) if not self.content[x] is None]
        ys = [self.content[x][0] for x in range(fitrange[0], fitrange[1]) if not self.content[x] is None]
        result = standard_fit(xs, ys, function, silent=True)
        [item.gamma_method() for item in result if isinstance(item,Obs)]
        return result

    #we want to quickly get a plateau
    def plateau(self, plateau_range, method="fit"):
        if self.N != 1:
            raise Exception("Correlator must be projected before getting a plateau.")
        if(all([self.content[t] is None for t in range(plateau_range[0], plateau_range[1])])):
                raise Exception("plateau is undefined at all timeslices in plateaurange.")
        if method == "fit":
            def const_func(a, t):
                return a[0] + a[1] * 0 # At some point pe.standard fit had an issue with single parameter fits. Being careful does not hurt
            return self.fit(const_func,plateau_range)[0]
        elif method in ["avg","average","mean"]:
            returnvalue= np.mean([item[0] for item in self.content if not item is None])
            returnvalue.gamma_method()
            return returnvalue

        else:
            raise Exception("Unsupported plateau method: " + method)

    #quick and dirty plotting function to view Correlator inside Jupyter
    #If one would not want to import pyplot, this could easily be replaced by a call to pe.plot_corrs
    #This might be a bit more flexible later
    def show(self,xrange=None,logscale=False):
        if self.N!=1:
            raise Exception("Correlator must be projected before plotting")
        if  xrange is None:
            xrange=[0,self.T]

        x,y,y_err=self.plottable()
        plt.errorbar(x,y,y_err,fmt="o-")
        if logscale:
            plt.yscale("log")
        else:
        # we generate ylim instead of using autoscaling.
            y_min=min([ (x[0].value-x[0].dvalue)  for x in self.content[xrange[0]:xrange[1]] if(not x is None)])
            y_max=max([ (x[0].value+x[0].dvalue)  for x in self.content[xrange[0]:xrange[1]] if(not x is None)])
            plt.ylim([y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min)])

        plt.xlabel(r"$n_t$ [a]")
        plt.xlim(xrange)
        plt.title("Quickplot")
        plt.grid()

        plt.show()
        plt.clf()
        return

    def dump(self,filename):
        dump_object(self,filename)
        return


    def __repr__(self):
        return("Corr[T="+str(self.T)+" , N="+str(self.N)+" , content="+str(self.content)+"]")
    def __str__(self):
        return ("Corr[T="+str(self.T)+" , N="+str(self.N)+" , content="+str(self.content)+"]")

    #We define the basic operations, that can be performed with correlators.
    #While */+- get defined here, they only work for Corr*Obs and not Obs*Corr.
    #This is because Obs*Corr checks Obs.__mul__ first and does not catch an exception.
    #One could try and tell Obs to check if the y in __mul__ is a Corr and

    def __add__(self, y):
        if isinstance(y, Corr):
            if ((self.N!=y.N) or (self.T!=y.T) ):
                raise Exception("Addition of Corrs with different shape")
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]+y.content[t])
            return Corr(newcontent)

        elif isinstance(y, Obs) or isinstance(y, int) or isinstance(y,float):
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]+y)
            return Corr(newcontent)
        else:
            raise TypeError("Corr + wrong type")

    def __mul__(self,y):
        if isinstance(y,Corr):
            if not((self.N==1 or y.N==1 or self.N==y.N) and self.T==y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]*y.content[t])
            return Corr(newcontent)

        elif isinstance(y, Obs) or isinstance(y, int) or isinstance(y,float):
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]*y)
            return Corr(newcontent)
        else:
            raise TypeError("Corr * wrong type")

    def __truediv__(self,y):
        if isinstance(y,Corr):
            if not((self.N==1 or y.N==1 or self.N==y.N) and self.T==y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]/y.content[t])
            #Here we set the entire timeslice to undefined, if one of the smearings has encountered an division by zero.
            #While this might throw away perfectly good values in other smearings, we will never have to check, if all values in our matrix are defined
            for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None

            if all([item is None for item in newcontent]):
                raise Exception("Division returns completely undefined correlator")



            return Corr(newcontent)

        elif isinstance(y, Obs):
            if y.value==0:
                raise Exception("Division by Zero will return undefined correlator")
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]/y)
            return Corr(newcontent)

        elif isinstance(y, int) or isinstance(y,float):
            if y==0:
                raise Exception("Division by Zero will return undefined correlator")
            newcontent=[]
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t]/y)
            return Corr(newcontent)
        else:
            raise TypeError("Corr / wrong type")

    def __neg__(self):
        newcontent=[None if (item is None) else -1.*item for item in self.content]
        return Corr(newcontent)

    def __sub__(self,y):
        return self +(-y)

    def __pow__(self, y):
        if isinstance(y, Obs) or isinstance(y,int) or isinstance(y,float):
            newcontent=[None if (item is None) else item**y for item in self.content]
            return Corr(newcontent)
        else:
            raise TypeError("type of exponent not supported")

    #The numpy functions:
    def sqrt(self):
        return self**0.5

    def log(self):
        newcontent=[None if (item is None) else np.log(item) for item in self.content]
        return Corr(newcontent)

    def exp(self):
        newcontent=[None if (item is None) else np.exp(item) for item in self.content]
        return Corr(newcontent)

    def sin(self):
        newcontent=[None if (item is None) else np.sin(item) for item in self.content]
        return Corr(newcontent)

    def cos(self):
        newcontent=[None if (item is None) else np.cos(item) for item in self.content]
        return Corr(newcontent)

    def tan(self):
        newcontent=[None if (item is None) else np.tan(item) for item in self.content]

        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")

        return Corr(newcontent)

    def sinh(self):
        newcontent=[None if (item is None) else np.sinh(item) for item in self.content]
        return Corr(newcontent)

    def cosh(self):
        newcontent=[None if (item is None) else np.cosh(item) for item in self.content]
        return Corr(newcontent)

    def tanh(self):
        newcontent=[None if (item is None) else np.tanh(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arcsin(self):
        newcontent=[None if (item is None) else np.arcsin(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arccos(self):
        newcontent=[None if (item is None) else np.arccos(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arctan(self):
        newcontent=[None if (item is None) else np.arctan(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arcsinh(self):
        newcontent=[None if (item is None) else np.arcsinh(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arccosh(self):
        newcontent=[None if (item is None) else np.arccosh(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)

    def arctanh(self):
        newcontent=[None if (item is None) else np.arctanh(item) for item in self.content]
        for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t]=None
        if all([item is None for item in newcontent]):
                raise Exception("Operation returns completely undefined correlator")
        return Corr(newcontent)


    #right hand side operations (require tweak in main module to work)
    def __rsub__(self,y):
        return -self+y
    def __rmul__(self, y):
        return self * y
    def __radd__(self,y):
        return self + y



##One of the most common tasks is to select a range for a plateau or a fit. This is best done visually.
#def GUI_range_finder(corr, current_range=None):
#    T=corr.T
#    if corr.N!=1:
#        raise Exception("The Corr needs to be projected to select a range.")
#    #We need to define few helper functions for the Gui
#    def get_figure(corr,values):
#        fig = matplotlib.figure.Figure(figsize=(7, 4), dpi=100)
#        fig.clf()
#        x,y,err=corr.plottable()
#        ax=fig.add_subplot(111,label="main")#.plot(t, 2 * np.sin(2 * np.pi * t))
#        end=int(max(values["range_start"],values["range_end"]))
#        start=int(min(values["range_start"],values["range_end"]))
#        db=[0.1,0.2,0.8]
#        ax.errorbar(x,y,err, fmt="-o",color=[0.4,0.6,0.8])
#        ax.errorbar(x[start:end],y[start:end],err[start:end], fmt="-o",color=db)
#        offset=int(0.3*(end-start))
#        xrange=[max(min(start-1,int(start-offset)),0),min(max(int(end+offset),end+1),T-1)]
#        ax.grid()
#        if values["Plateau"]:
#            plateau=corr.plateau([start,end])
#            ax.hlines(plateau.value,0,T+1,lw=plateau.dvalue,color="red",alpha=0.5)
#            ax.hlines(plateau.value,0,T+1,lw=1,color="red")
#            ax.set_title(r"Current Plateau="+str(plateau)[4:-1])
#        if(values["Crop X"]):
#            ax.set_xlim(xrange)
#        ax.set_xticks([x for x in ax.get_xticks() if (x-int(x)==0) and (0<=x<T)])
#        if(values["Crop Y"]):
#            y_min=min([ (x[0].value-x[0].dvalue)  for x in corr.content[xrange[0]:xrange[1]] if(not x is None)])
#            y_max=max([ (x[0].value+x[0].dvalue)  for x in corr.content[xrange[0]:xrange[1]] if(not x is None)])
#            ax.set_ylim([y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min)])
#        else:
#            y_min=min([ (x[0].value-x[0].dvalue)  for x in corr.content if(not x is None)])
#            y_max=max([ (x[0].value+x[0].dvalue)  for x in corr.content if(not x is None)])
#            ax.set_ylim([y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min)])
#        ax.vlines(values["range_start"]-0.5,-2*abs(y_min),2*y_max,color=db)
#        ax.vlines(values["range_end"]-0.5,-2*abs(y_min),2*y_max,color=db)
#        return fig
#
#    def draw_figure(canvas, figure):
#        #matplotlib.use('TkAgg')
#        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#        figure_canvas_agg.draw()
#        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
#        return figure_canvas_agg
#
#    def delete_figure_agg(figure_agg):
#        figure_agg.get_tk_widget().forget()
#        plt.close('all')
#
#    #We change settings for mpl only inside the function
#    #matplotlib.use('TkAgg')
#
#    #now we can call our gui
#    # define window layout
#    default_values={}
#    default_values["Crop X"]=False
#    default_values["Crop Y"]=False
#    default_values["Plateau"]=False
#    if current_range is None:
#        default_values["range_start"]=1
#        default_values["range_end"]=int(T/2)
#    else:
#        default_values["range_start"]=current_range[0]
#        default_values["range_end"]=current_range[1]
#
#
#    layout = [
#            [sg.Canvas(key='-CANVAS-')],
#            [sg.Slider(range=(0,T),default_value=default_values["range_start"],size=(40,15),orientation='horizontal',key="range_start",enable_events = True)],
#            [sg.Slider(range=(0,T),default_value=default_values["range_end"],size=(40,15),orientation='horizontal',key="range_end",enable_events = True)],
#            [sg.Checkbox('Crop X',key="Crop X",default=default_values["Crop X"],enable_events = True),sg.Checkbox('Crop Y',key="Crop Y",default=default_values["Crop Y"],enable_events = True),sg.Checkbox('Plateau', key="Plateau",default=default_values["Plateau"],enable_events = True),sg.Button('Return')]]
#
#    #Calling a theme after the layout is set, preserves default sliders and Buttons
#
#    window = sg.Window('Range Finder', layout, finalize=True, element_justification='center', font='Helvetica 18',return_keyboard_events=True)
#
#    # add the plot to the window
#    fig = get_figure(corr,default_values)
#    fig_canvas_agg =draw_figure(window['-CANVAS-'].TKCanvas, fig)
#    while True:
#        event, values = window.read()
#        if event is None or event=="Return" or event=="\r":
#            break
#        else:
#            if values["range_end"]<=values["range_start"]+2:
#                if values["range_start"]+3<T:
#                    window["range_end"].update(values["range_start"]+3)
#                else:
#                    window["range_start"].update(T-3)
#                    window["range_end"].update(values["range_start"]+3)
#            # we need a distance of 2 fo a plateau
#            if values["range_end"]<=values["range_start"]+1:
#                values["Plateau"]=False
#                window["Plateau"].update(False)
#            if fig_canvas_agg:
#                delete_figure_agg(fig_canvas_agg)
#            fig = get_figure(corr,values)
#            fig_canvas_agg =draw_figure(window['-CANVAS-'].TKCanvas, fig)
#
#
#    window.close()
#    #It is easier to read the last event, that occurred
#    if event=="Return" or event=="\r":
#        end=int(max(values["range_start"],values["range_end"]))
#        start=int(min(values["range_start"],values["range_end"]))
#        window.close()
#        return [start,end]
#    else:
#        return
