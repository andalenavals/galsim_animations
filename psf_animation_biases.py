import numpy as np
import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import galsim
#import galsim_hub

AUXNAME="deleteme.png"
NFRAMES=100
FIGZISE=(8,6)
stampsize=64
scale=0.7

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


def color_plot(x,y,z=None,colorlog=False,xtitle='',ytitle='',ctitle='',title=None, ftsize=16,xlim=None, ylim=None, cmap=None,filename=None, npoints_plot=None, linreg=False, xscalelog=False, yscalelog=False, yerr=None, s=10.0, alpha=0.9, alpha_err=0.1, scatterkwargs={}, sidehists=False, sidehistxkwargs={}, sidehistykwargs={}, colorbar=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import StrMethodFormatter, ScalarFormatter
    import random

    fig, ax= plt.subplots(figsize=FIGZISE)

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    scatterkwargs.update({"marker":"o", "alpha":alpha,"cmap":cmap,"s":s})

    if (npoints_plot is None): npoints_plot=len(x)
    if (npoints_plot> len(x)): npoints_plot=len(x)
    inds=random.sample([i for i in range(len(x))],npoints_plot)
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
        
        ret=linregfunc(x_unmask,y_unmask)
        m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
        ax.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='$\mu$: %.4f $\pm$ %.4f \n  c: %.4f $\pm$ %.4f'%(m,merr,c, cerr ))
        ax.legend(loc='upper left', prop={'size': ftsize-6})

    if colorlog:
        z=np.clip(abs(z), 1.e-8, 1.e28)
        colornorm=LogNorm( vmin=np.nanmin(z), vmax=np.nanmax(z))
        
        if ("vmin" in scatterkwargs)&("vmax" in scatterkwargs):
            colornorm=LogNorm( vmin=scatterkwargs["vmin"], vmax=scatterkwargs["vmax"])   
        scatterkwargs.update({"norm":colornorm, "vmin":None, "vmax":None}) 
    else:
        if ("vmin" in scatterkwargs)&("vmax" in scatterkwargs): scatterkwargs.update({"norm":None, "vmin":scatterkwargs["vmin"], "vmax":scatterkwargs["vmax"]})
        elif "norm" in scatterkwargs: scatterkwargs.update({"norm":scatterkwargs["norm"], "vmin":None, "vmax":None})
        else: scatterkwargs.update({"norm":None, "vmin":None, "vmax":None})
            
    if yerr is not None:
            ebarskwargs = {"fmt":'none', "color":"black", "ls":":", 'elinewidth':0.5, 'alpha':alpha_err}
            ax.errorbar(x[inds], y[inds], yerr=yerr, **ebarskwargs)
        
    if z is not None: scatterkwargs.update({"c":z[inds]})
    else: scatterkwargs.update({"c":None})
    sct=ax.scatter(x[inds], y[inds], **scatterkwargs)
    

    ax.set_xlabel(xtitle, fontsize=ftsize)
    ax.set_ylabel(ytitle, fontsize=ftsize)
    if title is not None: plt.title(title, fontsize=ftsize-4, loc='left', color='red')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xscalelog:
        plt.xscale('log')
    if yscalelog:
        if np.any(np.array(y[inds])!=0):
            plt.yscale('log')

    if (z is not None)& (colorbar==True):
        cbar=plt.colorbar(sct)
        cbar.ax.set_ylabel(ctitle, fontsize=ftsize-2, rotation=-90)
        cbar.ax.yaxis.set_label_coords(5.5,0.5)
   
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=200)

    plt.close()


def plots_psf_size():

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    sizemin=0.5
    sizemax=7.0
    d=(2*(sizemax-sizemin))/NFRAMES
    intervals=[0.5+i*d for i in range(int(NFRAMES//2))]
    intervals+=np.flip(intervals).tolist()
    
    nshears=50
    ngals=50
    shears=np.random.uniform(-0.1,0.1, nshears)
    for i in range(NFRAMES-1):    
        mean_obse=[]
        for s in shears:
            #make images
            gal_image = galsim.ImageF(stampsize, stampsize )
            psf_image = galsim.ImageF(stampsize, stampsize )
            galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

            # Make galaxy
            gal = galsim.Gaussian(sigma=5.0, flux=3.0)
            gal = gal.shear(g1=s, g2=0.0)
            image= gal.drawImage(gal_image, scale=scale)

            sigmalist=np.random.uniform(intervals[i],intervals[i+1], ngals)
            
            obse=[]
            for sigma in sigmalist:
                # Make PSF
                psf = galsim.Gaussian(sigma=sigma, flux=1.5)
                psf = psf.shear(g1=0.3, g2=0.0)
                psf.drawImage(psf_image, scale=scale)

                ##CONVOLVED
                galconv = galsim.Convolve([gal,psf])
                galconv.drawImage(galpsf_conv_image, scale=scale)
                
                #res = galsim.hsm.FindAdaptiveMom(galpsf_conv_image,  weight=None)
                #ada_g1 = res.observed_shape.g1
   

                hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                try:
                    result = galsim.hsm.EstimateShear(galpsf_conv_image, psf_image,  hsmparams=hsmparams, guess_sig_gal=5.0, guess_sig_PSF=sigma, shear_est="KSB")
                    ada_g1=result.corrected_g1
                except:
                    continue

                obse.append(ada_g1)
            mean_obse.append(np.mean(obse))

        comp=1
        xlabel=r"$g_{%i}$"%(comp)
        ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
        color_plot(shears,np.array(mean_obse), filename=AUXNAME, linreg=True, xtitle=xlabel,ytitle=ylabel, ylim=[-0.1,0.1], xlim=[-0.1,0.1])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=60
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('psf_size_bias.gif')
    plt.close(fig)
    

def plots_psf_anisotropy():

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    n1=NFRAMES//2
    d=2*np.pi/n1
    g1list=[0.8*np.sin(i*d) for i in range(n1)]
    g2list=[0.0]*n1
    
    g2list+=g1list
    g1list+=[0.0]*n1
    
    nshears=50
    ngals=50
    shears=np.random.uniform(-0.1,0.1, nshears)
    for i in range(NFRAMES-1):    
        mean_obse=[]
        for s in shears:
            #make images
            gal_image = galsim.ImageF(stampsize, stampsize )
            psf_image = galsim.ImageF(stampsize, stampsize )
            galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

            # Make galaxy
            gal = galsim.Gaussian(sigma=5.0, flux=2.5)
            gal = gal.shear(g1=s, g2=0.0)
            image= gal.drawImage(gal_image, scale=scale)

            g1listaux=np.random.uniform(g1list[i],g1list[i+1], ngals)
            g2listaux=np.random.uniform(g2list[i],g2list[i+1], ngals)
            
            obse=[]
            for g1,g2 in zip(g1listaux,g2listaux):
                # Make PSF
                psf = galsim.Gaussian(sigma=2.5, flux=0.8)
                psf = psf.shear(g1=g1, g2=g2)
                psf.drawImage(psf_image, scale=scale)

                ##CONVOLVED
                galconv = galsim.Convolve([gal,psf])
                galconv.drawImage(galpsf_conv_image, scale=scale)
                
                #res = galsim.hsm.FindAdaptiveMom(galpsf_conv_image,  weight=None)
                #ada_g1 = res.observed_shape.g1

                try:
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    result = galsim.hsm.EstimateShear(galpsf_conv_image, psf_image,  hsmparams=hsmparams, guess_sig_gal=5.0, guess_sig_PSF=2.5, shear_est="KSB")
                    ada_g1=result.corrected_g1
                except:
                    continue
                
                obse.append(ada_g1)
            mean_obse.append(np.mean(obse))
        comp=1
        xlabel=r"$g_{%i}$"%(comp)
        ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
        #color_plot(shears,np.array(mean_obse), filename="%i.png"%(i), linreg=True, xtitle=xlabel,ytitle=ylabel, ylim=[-0.3,0.3], xlim=[-0.1,0.1])
        color_plot(shears,np.array(mean_obse), filename=AUXNAME, linreg=True, xtitle=xlabel,ytitle=ylabel, ylim=[-0.3,0.3], xlim=[-0.1,0.1])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    speed=60
    ani = anime.ArtistAnimation(fig, ims, interval=speed*(100/NFRAMES), blit=True)
    ani.save('psf_anisotropy_bias.gif')
    plt.close(fig)
   

def main():
    print("Doing PSF size")
    plots_psf_size()
    print("Doing PSF anisotropy")
    plots_psf_anisotropy()

    

if __name__ == "__main__":
    main()
