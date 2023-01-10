import galsim
#import galsim_hub
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import astropy.visualization
from astropy.table import Table


AUXNAME="deleteme.png"
NFRAMES=100
FIGZISE=(14,4)
stampsize=64
scale=0.7
method="ksb" #"hsm"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PSF biases')
    parser.add_argument('--workdir',
                        default='out', 
                        help='diractory of work')   
    args = parser.parse_args()

    return args

def make_plot(galimg, psfimg, galpsfimg,x,y,plotname='test.png', vmin=None, vmax=None,ylim=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np
    import pandas as pd

    fig, axs= plt.subplots(1, 4, figsize=FIGZISE)
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax0=axs[0]; ax1=axs[1]; ax2=axs[2]
    arrays=[img.array for img in [galimg, psfimg, galpsfimg]]
    cmap='gray'
    norm=None
    
    for ax, dat,title in zip([ax0,ax1,ax2],arrays,["Galaxy","PSF", "Observed galaxy"]):
        #vmin=None; vmax=None
        ax.imshow(dat,cmap=cmap,norm=norm, vmin=vmin, vmax=vmax)
        ax.set_ylim(0,stampsize-1)
        ax.set_title(title,size=14, color="yellow")
        ax.patch.set_edgecolor('white')
        ax.patch.set_linewidth('2') 
        #ax.axis('off')

    comp=1
    xlabel=r"$g_{%i}$"%(comp)
    ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
    color_plot_ax(axs[3],x,y,z=None,colorlog=False,xtitle=xlabel,ytitle=ylabel,ctitle="",ftsize=16,xlim=[-0.1,0.1], ylim=ylim,cmap=None,filename=None, colorbar=False,linreg=True)
    axs[2].set_title(title,size=14, color="yellow")

    fig.tight_layout()
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gift
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


def plots_psf_size():
    #make images
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

   
    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]

    d=np.pi/NFRAMES
    sigmalist=[0.5+7.0*np.sin(d*i) for i in range(NFRAMES)]

    nshears=50
    ngals=5
    shears=np.random.uniform(-0.1,0.1, nshears)
    
    for i, sigma in enumerate(sigmalist):
        print("Doing frame %i"%(i))
        mean_obse=[]
        for s in shears:
            obse=[]
            for n in range(ngals):
                # Make galaxy
                gal = galsim.Gaussian(sigma=5.0, flux=3.0)
                #gal = gal.shear(g1=np.random.uniform(-0.1,0.1), g2=0.0)
                #gal = gal.shear(g1=0.0, g2=np.random.uniform(-0.1,0.1))
                gal = gal.shear(g1=0.0, g2=0.5)
                gal = gal.shear(g1=s, g2=0.0)
                gal.drawImage(gal_image, scale=scale)
                
                # Make PSF
                psf = galsim.Gaussian(sigma=sigma, flux=1.5)
                #psf = psf.shear(g1=0.2, g2=0.1)
                psf.drawImage(psf_image, scale=scale)

                ##CONVOLVED
                galconv = galsim.Convolve([gal,psf])
                galconv.drawImage(galpsf_conv_image, scale=scale)

                if method =="hsm":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        res = galsim.hsm.FindAdaptiveMom(galpsf_conv_image,  weight=None, hsmparams=hsmparams)
                        ada_g1 = res.observed_shape.g1
                    except:
                        print("Failed %i galaxy"%(n))
                        continue
                
                elif method=="ksb":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        result = galsim.hsm.EstimateShear(galpsf_conv_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB")
                        ada_g1=result.corrected_g1
                    except:
                        continue
                else:
                    print("No measurement method defined")
                    raise 
                
                obse.append(ada_g1)
            mean_obse.append(np.mean(obse))

        biases=np.array(mean_obse)-shears
        make_plot(gal_image, psf_image, galpsf_conv_image, shears,biases, plotname=AUXNAME, vmin=0, vmax=0.015, ylim=[-0.1,0.1])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ani = anime.ArtistAnimation(fig, ims, interval=60, blit=True)
    ani.save('psf_size.gif')
    plt.close(fig)

def plots_psf_anisotropy():
    catalog = Table([[5., 3. ,2.], [29., 20., 15.], [0.0, 0.2, 0.5] ],
             names=['flux_radius', 'mag_auto', 'zphot'])
    
    #make images
    
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

    
    # Make galaxy
    #model = galsim_hub.GenerativeGalaxyModel('https://zenodo.org/record/7457343/files/model.tar.gz')
    #profiles = model.sample(catalog)
    #gal=profiles[0]
    #gal.drawImage(gal_image, scale=0.001)
    gal = galsim.Gaussian(sigma=5.0, flux=2.5)
    gal = gal.shear(g1=0.5, g2=0.0)
    image= gal.drawImage(gal_image, scale=scale)

   
    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    n1=NFRAMES//2
    n2=NFRAMES-n1

    d=2*np.pi/n1
    g1list=[0.8*np.sin(i*d) for i in range(n1)]
    g2list=[0.0]*n1
    
    g2list+=g1list
    g1list+=[0.0]*n1
    
    nshears=50
    ngals=2
    shears=np.random.uniform(-0.1,0.1, nshears)
    i=0
    for g1,g2 in zip(g1list,g2list):
        print("Doing frame %i"%(i))
        mean_obse=[]
        for s in shears:
            obse=[]
            for n in range(ngals):
                 # Make galaxy
                gal = galsim.Gaussian(sigma=5.0, flux=2.5)
                #gal = gal.shear(g1=np.random.uniform(-0.1,0.1), g2=0.0)
                #gal = gal.shear(g1=0.0, g2=np.random.uniform(-0.1,0.1))
                gal = gal.shear(g1=0.0, g2=0.5)
                gal = gal.shear(g1=s, g2=0.0)
                image= gal.drawImage(gal_image, scale=scale)
                # Make PSF
                psf = galsim.Gaussian(sigma=1.5, flux=0.8)
                psf = psf.shear(g1=g1, g2=g2)
                psf.drawImage(psf_image,scale=scale)

                ##CONVOLVED
                galconv = galsim.Convolve([gal,psf])
                galconv.drawImage(galpsf_conv_image,scale=scale)

                if method =="hsm":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        res = galsim.hsm.FindAdaptiveMom(galpsf_conv_image,  weight=None, hsmparams=hsmparams)
                        ada_g1 = res.observed_shape.g1
                    except:
                        print("Failed %i galaxy"%(n))
                        continue
                
                elif method=="ksb":
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        result = galsim.hsm.EstimateShear(galpsf_conv_image, psf_image,  hsmparams=hsmparams,   shear_est="KSB")
                        ada_g1=result.corrected_g1
                    except:
                        continue
                else:
                    print("No measurement method defined")
                    raise
                
                obse.append(ada_g1)
            mean_obse.append(np.mean(obse))

        biases=np.array(mean_obse)-shears

        make_plot(gal_image, psf_image, galpsf_conv_image,shears, biases, plotname=AUXNAME, vmin=0, vmax=0.015, ylim=[-0.2,0.2])
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])
        i+=1

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ani = anime.ArtistAnimation(fig, ims, interval=60, blit=True)
    ani.save('psf_anisotropy.gif')
    plt.close(fig)
   
def main():
    print("Doing PSF size")
    plots_psf_size()
    print("Doing PSF anisotropy")
    plots_psf_anisotropy()

    

if __name__ == "__main__":
    main()
