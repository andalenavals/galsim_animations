import galsim
import numpy as np
import random
import scipy
import scipy.stats
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import astropy.visualization
import datetime
import multiprocessing
import time

AUXNAME="deleteme5.png"
NFRAMES=20
NREP=10
FIGZISE=(8,3)
imagesize=600 # 600 pix = 1arcmin for 0.1 pixescale
scale=0.1
psfpixelscale = 0.02 #arcsec
method=["hsm","ksb"][0]
sigmanoise=0

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PSF biases')
    parser.add_argument('--workdir',
                        default='out', 
                        help='diractory of work')   
    args = parser.parse_args()

    return args

def linregfunc(x, y, prob=0.68):
	"""
	A linear regression y = m*x + c, with confidence intervals on m and c.
	
	As a safety measure, this function will refuse to work on masked arrays.
	Indeed scipy.stats.linregress() seems to silently disregard masks...
	... and as a safety measure, we compare against scipy.stats.linregress().
	
	"""
	
	if len(x) != len(y):
		raise RuntimeError("Your arrays x and y do not have the same size")
	
	if np.ma.is_masked(x) or np.ma.is_masked(y):
		raise RuntimeError("Do not give me masked arrays")
	
	n = len(x)
	xy = x * y
	xx = x * x
	
	b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
	b0 = y.mean() - b1 * x.mean()
	
	#s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in xrange(n)])
	s2 = np.sum((y - b0 - b1 * x)**2) / n
	
	alpha = 1.0 - prob
	c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
	c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
	#print 'the confidence interval of s2 is: ',[n*s2/c2,n*s2/c1]
	
	c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
	bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
	
	bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
	
	ret = {"m":b1-1.0, "c":b0, "merr":bb1, "cerr":bb0}
	
	# A little test (for recent numpy, one would use np.isclose() for this !)
	#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	'''
	if not abs(slope - b1) <= 1e-6 * abs(slope):
		raise RuntimeError("Slope error, %f, %f" % (slope, b1))
	if not abs(intercept - b0) <= 1e-6 * abs(intercept):
		raise RuntimeError("Intercept error, %f, %f" % (intercept, b0))
	'''
	return ret

def linregw(x,y,w, sigma_shape=0.0):
        from scipy.optimize import curve_fit
        #sigma = sigma_shape * np.sqrt(1.0/np.clip(w, 1e-18, 1e18))# if w means selection
        sigma = np.sqrt(sigma_shape**2 + 1.0/np.clip(w, 1e-18, 1e18))

        absolute_sigma=True # True#False

        def f(x, a, b): return a * x + b

        p0 = [1.0, 0.0] # initial parameter estimate
        popt, pcov = curve_fit(f, x, y, p0, sigma, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))
        
        m = popt[0] - 1.0
        c = popt[1]
        merr = perr[0]
        cerr = perr[1]
        ret = {"m":m, "c":c, "merr":merr, "cerr":cerr}
        return ret 

def make_plot(galimg,x,y,yerr=None, plotname='test.png', vmin=None, vmax=None, title=""):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, axs= plt.subplots(1, 2, figsize=FIGZISE)
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    cmap='gray'
    
    #vmin=None; vmax=None
    #divider = make_axes_locatable(axs[0])
    #cax = divider.append_axes('right', size='5%', pad=0.05)

    #im=axs[0].imshow(dat,cmap=cmap,norm=LogNorm( vmin=vmin, vmax=vmax))
    im=axs[0].imshow(galimg.array,cmap=cmap,norm=None, vmin=vmin, vmax=vmax)
    axs[0].set_ylim(0,imagesize-1)
    axs[0].set_title(title,size=14, color="yellow")
    axs[0].patch.set_edgecolor('white')
    axs[0].patch.set_linewidth('2')
    axs[0].text(imagesize//2, -imagesize//8, r'1 arcmin$^{2}$', horizontalalignment='center', verticalalignment='center', color="yellow", animated=True, size=14)

    
    #plt.colorbar(im,cax=cax)

    comp=1
    xlabel=r"$g_{%i}$"%(comp)
    ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
    color_plot_ax(axs[1],x,y,yerr=yerr,z=None,colorlog=False,xtitle=xlabel,ytitle=ylabel,ctitle="",ftsize=16,xlim=[-0.1,0.1], ylim=[-0.25,0.25],cmap=None,filename=None, colorbar=False,linreg=True)
    axs[1].set_title(title,size=14, color="yellow")
    
    #ax.axis('off')
    fig.tight_layout()
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gift
    plt.close(fig)

def color_plot_ax(ax,x,y,z=None,yerr=None,colorlog=True,xtitle="",ytitle="",ctitle="",ftsize=16,xlim=None, ylim=None,cmap=None,filename=None, colorbar=True,linreg=True):
    "if linreg is used y must be biases"
    if linreg:
        xplot=np.linspace(min(x),max(x))
        #mask=~x.mask&~y.mask
        #x_unmask,y_unmask=x[mask].data,y[mask].data
        are_maskx=type(x)==np.ma.masked_array
        are_masky=type(y)==np.ma.masked_array
        are_masked=(np.ma.is_masked(x))|(np.ma.is_masked(y))|(are_maskx|are_masky)
        if are_masked:
            mask=~x.mask&~y.mask
            x_unmask,y_unmask=x[mask].data,y[mask].data
        else:
            x_unmask,y_unmask=x,y
        
        if yerr is not None:
            ret=linregw(x_unmask,y_unmask,1./(yerr**2))
        else:
            ret=linregfunc(x_unmask,y_unmask)
        m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
        ax.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='$\mu_{1}$: %.4f $\pm$ %.4f \n  c$_{1}$: %.4f $\pm$ %.4f'%(m,merr,c, cerr ))
        ax.legend(loc='upper left', prop={'size': ftsize-4}, labelcolor="yellow")
        
    if colorlog: 
        c=abs(c)
        colornorm=LogNorm( vmin=np.nanmin(c), vmax=np.nanmax(c))
    else: colornorm=None


    sct=ax.scatter(x, y,c=z, norm=colornorm, marker=".",alpha=0.7,cmap=cmap)

    if yerr is not None:
            ebarskwargs = {"fmt":'none', "color":"yellow", "ls":":", 'elinewidth':1.5, 'alpha':1.0}
            ax.errorbar(x, y, yerr=yerr, **ebarskwargs)

    
    ax.set_xlabel(xtitle, fontsize=ftsize)
    ax.set_ylabel(ytitle, fontsize=ftsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if colorbar:
        cbar=plt.colorbar(sct, ax=ax)
        cbar.ax.set_xlabel(ctitle, fontsize=ftsize-2)
        cbar.ax.xaxis.set_label_coords(0.5,1.1)

    for pos in ["left","right","top","bottom"]:
        ax.spines[pos].set_color('white')
    ax.yaxis.label.set_color('yellow')
    ax.xaxis.label.set_color('yellow')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('black')

def plot_neighbors():
    
    psf_image = galsim.ImageF(imagesize, imagesize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    #fmin=10
    #fmax=70
    #df=2*(fmax-fmin)/NFRAMES
    #density_list=np.concatenate([[fmin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()

    #density_list=[10,10,10,20,20,20,30,30,30,40,40,40, 50, 50, 50, 60, 60, 60, 70, 70, 70]
    #density_list=[10,20,30,40,50,60,70,80]
    density_list=[1,2,3]
    density_list+=np.flip(density_list).tolist()

    vmin=0
    vmax=None #2000

    print(density_list)

    nshears=50
    nimages=50
    shears=np.random.uniform(-0.1,0.1, nshears)
    
    rng = galsim.BaseDeviate(123)
    #rng = None
    for d in density_list:
        mean_obse=[]; std_obse=[]
        print("doing %i galaxies"%(d))
        for s in shears:
            mean_obse_img=[]; std_obse_img=[]
            for img in range(nimages):
                gal_image = galsim.ImageF(imagesize, imagesize )
                poslist=[]
                for j in range(int(d)):
                    x=np.random.uniform(0,imagesize)
                    y=np.random.uniform(0,imagesize)
                    pos = galsim.PositionD(x,y)
                    poslist.append(pos)
            
                    gal = galsim.Gaussian(sigma=.5, flux=1.e2)
                    gal = gal.shear(g1=0, g2=0.5)
                    gal = gal.shear(g1=s, g2=0.0)
                    psf = galsim.Gaussian(sigma=3.55*psfpixelscale, flux=1.0)
                    psf.drawImage(psf_image, scale=scale)
                    #psf = psf.shear(g1=0.2, g2=0.1)
                    galconv = galsim.Convolve([gal,psf])
                    galconv.drawImage(gal_image, center=pos, scale=scale, add_to_image=True)

                obse=[]
                for pos in poslist:
                    if method =="hsm":
                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                        try:
                            res = galsim.hsm.FindAdaptiveMom(gal_image,  weight=None, hsmparams=hsmparams, guess_centroid=pos)
                            ada_g1 = res.observed_shape.g1
                        except:
                            print("Failed galaxy")
                            continue
                
                    elif method=="ksb":
                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                        try:
                            result = galsim.hsm.EstimateShear(gal_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB", guess_centroid=pos)
                            ada_g1=result.corrected_g1
                        except:
                            print("Failed galaxy")
                            continue
                    else:
                        print("No measurement method defined")
                        raise
                    obse.append(ada_g1)
                mean_obse_img.append(np.mean(obse))

            mean_obse.append(np.mean(mean_obse_img))
            std_obse.append(np.std(mean_obse_img))
        

        biases=np.array(mean_obse)-shears
        make_plot(gal_image, shears,biases, plotname=AUXNAME, vmin=vmin, vmax=vmax, title="n = %i"%(d))
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

        #vminaux,vmaxaux=vis.get_limits(gal_image.array)
        #print(np.max(gpoisson_image.array))
        #print(flux_list[i], np.max(gal_image.array), vminaux, vmaxaux)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=100
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('neigbors.gif')
    plt.close(fig)


def plot_neighbors_parallel():
    psf_image = galsim.ImageF(imagesize, imagesize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    #fmin=10
    #fmax=70
    #df=2*(fmax-fmin)/NFRAMES
    #density_list=np.concatenate([[fmin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()

    #density_list=[10,10,10,20,20,20,30,30,30,40,40,40, 50, 50, 50, 60, 60, 60, 70, 70, 70]
    #density_list=[240,10]
    density_list=[10,30,60,90,180]
    density_list+=np.flip(density_list).tolist()

    #nimages=[5, 120]
    nimages=[180,60,30,20,10]
    nimages+=np.flip(nimages).tolist()

    #gridlist=[False,True]
    #gridlist+=np.flip(gridlist).tolist()
    gridlist=[False]*len(nimages)
    

    vmin=0
    vmax=2

    print(density_list)

    nshears=50
    #nimages=120
    shears=np.random.uniform(-0.1,0.1, nshears)

    
    rng = galsim.BaseDeviate(123)

    #rng = None
    for d, nimg,grid in zip(density_list,nimages,gridlist):
        mean_obse=[]; std_obse=[]
        print("doing %i galaxies"%(d))

        gal_image = galsim.ImageF(imagesize, imagesize )

        for j in range(int(d)):
            if grid:
                x,y=draw_grid(imagesize, idx=j, ngals=d)
            else:
                x=np.random.uniform(0,imagesize)
                y=np.random.uniform(0,imagesize)
            pos = galsim.PositionD(x,y)
            gal = galsim.Gaussian(sigma=.5, flux=5.e2)
            gal = gal.shear(g1=0, g2=np.random.choice([-0.5,0.5]))
            
            #gal = gal.shear(g1=shear, g2=0.0)
            psf = galsim.Gaussian(sigma=3.55*psfpixelscale, flux=1.0)
            #psf.drawImage(psf_image, scale=scale)
            #psf = psf.shear(g1=0.2, g2=0.1)
            galconv = galsim.Convolve([gal,psf])
            galconv.drawImage(gal_image, center=pos, scale=scale, add_to_image=True)

        gaussian_noise = galsim.GaussianNoise(rng, sigma=sigmanoise)
        gal_image.addNoise(gaussian_noise)
        
        for s in shears:
            pool = multiprocessing.Pool(processes=nimg)
            all_obse_img=pool.map(worker, [{"ngals":d,"shear":s, "grid":grid}]*nimg)
            all_obse_img=np.concatenate(all_obse_img)
            pool.close()
            pool.join()

                  
            #q=multiprocessing.Queue()
            #p=multiprocessing.Process(target=worker, args=(q, imgid ))
            #p.start()
            #print("Getting prediction in Queue")
            #preds=q.get()
            #p.join()
                
            mean_obse.append(np.mean(all_obse_img))
            std_obse.append(np.std(all_obse_img))
        

        biases=np.array(mean_obse)-shears
        make_plot(gal_image, shears,biases,yerr=np.array(std_obse), plotname=AUXNAME, vmin=vmin, vmax=vmax, title="n = %i"%(d))
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

        #vminaux,vmaxaux=vis.get_limits(gal_image.array)
        #print(np.max(gpoisson_image.array))
        #print(flux_list[i], np.max(gal_image.array), vminaux, vmaxaux)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=100
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('neigbors.gif')
    plt.close(fig)


def draw_grid(imagesize, idx=None, ngals=None):   
    aux=np.sqrt(ngals)
    if int(aux)==aux:
        nparts_side= aux
    else:
        nparts_side= int(aux) +1
            
    (piy, pix) = divmod(idx, nparts_side)
    stampsize=imagesize/nparts_side
    x=(pix+0.5)*stampsize
    y=(piy+0.5)*stampsize

    return x, y
    
def worker(args):
    ngals=int(args["ngals"])
    shear=args["shear"]
    grid=args["grid"]

    starttime = datetime.datetime.now()
    np.random.seed()
    #random.seed()
    p = multiprocessing.current_process()
    print("%s is starting with PID %s" % (p.name,  p.pid))

    rng = galsim.BaseDeviate(123)
    gal_image = galsim.ImageF(imagesize, imagesize )
    psf_image = galsim.ImageF(imagesize, imagesize )
    poslist=[]
    for j in range(ngals):
        if grid:
            x,y=draw_grid(imagesize, idx=j, ngals=ngals)
        else:
            x=np.random.uniform(0,imagesize)
            y=np.random.uniform(0,imagesize)
        pos = galsim.PositionD(x,y)
        poslist.append(pos)
        
        gal = galsim.Gaussian(sigma=.5, flux=5.e2)
        gal = gal.shear(g1=0, g2=np.random.choice([-0.5,0.5]))
        gal = gal.shear(g1=shear, g2=0.0)
        psf = galsim.Gaussian(sigma=3.55*psfpixelscale, flux=1.0)
        psf.drawImage(psf_image, scale=scale)
        #psf = psf.shear(g1=0.2, g2=0.1)
        galconv = galsim.Convolve([gal,psf])
        galconv.drawImage(gal_image, center=pos, scale=scale, add_to_image=True)
    gaussian_noise = galsim.GaussianNoise(rng, sigma=sigmanoise)
    gal_image.addNoise(gaussian_noise)

    obse=[]
    failed=0
    #print(ngals, len(poslist))
    #assert False
    for pos in poslist:
        if method =="hsm":
            '''
            measstamp=32
            x=pos.x
            y=pos.y
            lowx=int(np.floor(x-0.5*measstamp))
            lowy=int(np.floor(y-0.5*measstamp))
            upperx=int(np.floor(x+0.5*measstamp))
            uppery=int(np.floor(y+0.5*measstamp))
            if lowx < 1 :flag=1 ;lowx=1
            if lowy < 1 :flag=1 ;lowy=1
            if upperx > imagesize : flag=1; upperx =imagesize
            if uppery > imagesize : flag=1; uppery =imagesize
            bounds = galsim.BoundsI(lowx,upperx , lowy , uppery )
            gps = gal_image[bounds]
            '''
            
            hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
            try:
                res = galsim.hsm.FindAdaptiveMom(gal_image,  weight=None, hsmparams=hsmparams, guess_centroid=pos, guess_sig=5,)
                ada_g1 = res.observed_shape.g1
            except Exception as e:
                print(e)
                failed+=1
                continue
        
        elif method=="ksb":
            hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
            try:
                result = galsim.hsm.EstimateShear(gal_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB", guess_centroid=pos, sky_var=sigmanoise**2)
                ada_g1=result.corrected_g1
            except:
                failed+=1
                #print("Failed galaxy")
                continue
        else:
            print("No measurement method defined")
            raise
        obse.append(ada_g1)
    #q.put(np.mean(obse)))
    
    endtime = datetime.datetime.now()
    print("%s is done, it took %s. Worker with density %i, failed %i" % (p.name, str(endtime - starttime), ngals, failed))
    #return np.mean(obse)
    return obse
    

def main():
    #plot_neighbors()
    plot_neighbors_parallel()

if __name__ == "__main__":
    main()
