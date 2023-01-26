import galsim
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import astropy.visualization

AUXNAME="deleteme2.png"
NFRAMES=100
NREP=2
FIGZISE=(15,4)
stampsize=64
scale=0.7
method=["hsm","ksb"][0]


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

        absolute_sigma=False

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

def make_plot(galimg, galimgnoise, x, y, yerr, plotname='test.png', vmin=None, vmax=None, title="", ylim=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, axs= plt.subplots(1, 3, figsize=FIGZISE)
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    cmap='gray'
    
    #vmin=None; vmax=None
    #for ax, dat,title,vmin,vmax in zip([ax0,ax1],arrays,["Galaxy","Poisson Noise"],[vmin,None],[vmax,None]):
    #for ax, dat,tit in zip([ax0,ax1],arrays,["Galaxy",title]):

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #im=axs[0].imshow(galimg.array,cmap=cmap,norm=LogNorm( vmin=vmin, vmax=vmax))
    im=axs[0].imshow(galimg.array,cmap=cmap,norm=None, vmin=vmin, vmax=vmax)
    axs[0].set_ylim(0,stampsize-1)
    axs[0].set_title("Galaxy",size=14, color="yellow")
    axs[0].patch.set_edgecolor('white')
    axs[0].patch.set_linewidth('2')
    #plt.colorbar(im,cax=cax)
    #ax.axis('off')

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #im=axs[1].imshow(galimgnoise.array,cmap=cmap,norm=LogNorm( vmin=vmin, vmax=vmax))
    #im=axs[1].imshow(galimgnoise.array,cmap=cmap,norm=None, vmin=vmin, vmax=vmax)
    im=axs[1].imshow(galimgnoise.array,cmap=cmap,norm=None, vmin=None, vmax=None)
    axs[1].set_ylim(0,stampsize-1)
    axs[1].set_title(title,size=14, color="yellow")
    axs[1].patch.set_edgecolor('white')
    axs[1].patch.set_linewidth('2')
    #plt.colorbar(im,cax=cax)
    #ax.axis('off')

    comp=1
    xlabel=r"$g_{%i}$"%(comp)
    ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
    color_plot_ax(axs[2],x,y,yerr=yerr, z=None,colorlog=False,xtitle=xlabel,ytitle=ylabel,ctitle="",ftsize=16,xlim=[-0.1,0.1], ylim=ylim,cmap=None,filename=None, colorbar=False,linreg=True)
    #axs[2].set_title(title,size=14, color="yellow")

    fig.tight_layout()
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gif
    plt.close(fig)

def color_plot_ax(ax,x,y,z=None,yerr=None,colorlog=True,xtitle="",ytitle="",ctitle="",ftsize=16,xlim=None, ylim=None,cmap=None,filename=None, colorbar=True,linreg=True):

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


    



def plots_poisson_noise():    
    #make images
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    gpoisson_image = galsim.ImageF(stampsize, stampsize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    #flux_list=[100+2.3**i for i in range(int(NFRAMES//2))]
    fmin=1.e3
    fmax=5.e5
    df=2*(fmax-fmin)/NFRAMES
    flux_list=np.concatenate([[fmin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()
    flux_list=[fmin]*5+flux_list
    flux_list+=np.flip(flux_list).tolist()

    vmin=0
    vmax=500
    
    #vis=astropy.visualization.ZScaleInterval(nsamples=1000, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5, max_iterations=5)
    
    rng = galsim.BaseDeviate(123)

    nshears=50
    ngals=50
    shears=np.random.uniform(-0.1,0.1, nshears)
    
    for e,i in enumerate(range(len(flux_list))):
        print("Doing frame %i"%(e))
        mean_obse=[]; std_obse=[]
        for s in shears:
            obse=[]
            for n in range(ngals):
                gal = galsim.Gaussian(sigma=5.0, flux=flux_list[i])
                gal = gal.shear(g1=0.0, g2=0.5)
                gal = gal.shear(g1=s, g2=0.0)
                psf = galsim.Gaussian(sigma=2.5, flux=1.0)
                #psf = psf.shear(g1=0.2, g2=0.1)
                psf.drawImage(psf_image, scale=scale)
                galconv = galsim.Convolve([gal,psf])
                galconv.drawImage(gal_image, scale=scale)
                galconv.drawImage(gpoisson_image, scale=scale)
        
                poisson_noise = galsim.PoissonNoise(rng, sky_level=0.)
                gpoisson_image.addNoise(poisson_noise)

                if method =="hsm":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        res = galsim.hsm.FindAdaptiveMom(gpoisson_image,  weight=None, hsmparams=hsmparams)
                        ada_g1 = res.observed_shape.g1
                    except:
                        print("Failed %i galaxy"%(n))
                        continue
                
                elif method=="ksb":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        result = galsim.hsm.EstimateShear(gpoisson_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB")
                        ada_g1=result.corrected_g1
                    except:
                        continue
                else:
                    print("No measurement method defined")
                    raise
                
                obse.append(ada_g1)

            mean_obse.append(np.mean(obse)); std_obse.append(np.std(obse))
                
        biases=np.array(mean_obse)-shears
        make_plot(gal_image, gpoisson_image,shears,biases, np.array(std_obse), plotname=AUXNAME, vmin=vmin, vmax=vmax, title="Poisson noise", ylim=[-0.1,0.1])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

        #vminaux,vmaxaux=vis.get_limits(gal_image.array)
        #print(np.max(gpoisson_image.array))
        #print(flux_list[i], np.max(gal_image.array), vminaux, vmaxaux)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=100
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('poisson_noise.gif')
    plt.close(fig)


def plots_gaussian_noise():   
    #make images
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    ggaussian_image = galsim.ImageF(stampsize, stampsize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    smin=1.e0
    smax=1.e4
    df=2*(smax-smin)/NFRAMES

    sigmalist=np.concatenate([[smin+i*df]*NREP for i in range(int(NFRAMES//(2)))]).tolist()
    sigmalist+=np.flip(sigmalist).tolist()

    vmin=0
    vmax=3.e4
    
    rng = galsim.BaseDeviate(123)
    nshears=50
    ngals=50
    shears=np.random.uniform(-0.1,0.1, nshears)

    
    for i, sigma in enumerate(sigmalist):
        print("Doing frame %i, %.2f"%(i, sigma))
        mean_obse=[]; std_obse=[]
        for s in shears:
            obse=[]
            for n in range(ngals):
                gal = galsim.Gaussian(sigma=5.0, flux=1.e7)
                gal = gal.shear(g1=0.0, g2=0.5)
                gal = gal.shear(g1=s, g2=0.0)
                psf = galsim.Gaussian(sigma=2.5, flux=1.0)
                #psf = psf.shear(g1=0.2, g2=0.1)
                galconv = galsim.Convolve([gal,psf])
                psf.drawImage(psf_image, scale=scale)
                galconv.drawImage(gal_image, scale=scale)
                galconv.drawImage(ggaussian_image, scale=scale)
        
                gaussian_noise = galsim.GaussianNoise(rng, sigma=sigma)
                ggaussian_image.addNoise(gaussian_noise)

                if method =="hsm":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        res = galsim.hsm.FindAdaptiveMom(ggaussian_image,  weight=None, hsmparams=hsmparams)
                        ada_g1 = res.observed_shape.g1
                    except:
                        print("Failed %i galaxy"%(n))
                        continue
                
                elif method=="ksb":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        result = galsim.hsm.EstimateShear(ggaussian_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB")
                        ada_g1=result.corrected_g1
                    except:
                        continue
                else:
                    print("No measurement method defined")
                    raise
                
                obse.append(ada_g1)

            mean_obse.append(np.mean(obse))
            std_obse.append(np.std(obse))

        biases=np.array(mean_obse)-shears
        make_plot(gal_image, ggaussian_image, shears, biases, yerr=np.array(std_obse), plotname=AUXNAME, vmin=vmin, vmax=vmax,title="Galaxy + read noise",  ylim=[-0.1,0.1])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])


        #vis=astropy.visualization.ZScaleInterval(nsamples=1000, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5, max_iterations=5)
        #vminaux,vmaxaux=vis.get_limits(gal_image.array)
        #print(np.max(ggaussian_image.array))
        #print(flux_list[i], np.max(gal_image.array), vminaux, vmaxaux)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=100
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('gaussian_noise.gif')
    plt.close(fig)


def main():
    print("Plotting Poisson noise")
    plots_poisson_noise()
    print("Plotting Gaussian noise")
    #plots_gaussian_noise()



    
if __name__ == "__main__":
    main()
