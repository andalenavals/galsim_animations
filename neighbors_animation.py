import galsim
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import astropy.visualization

AUXNAME="deleteme5.png"
NFRAMES=20
NREP=10
FIGZISE=(8,3)
stampsize=600 # 600 pix = 1arcmin for 0.1 pixescale
scale=0.1
psfpixelscale = 0.02 #arcsec

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

def make_plot(galimg,x,y,plotname='test.png', vmin=None, vmax=None, title=""):
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
    axs[0].set_ylim(0,stampsize-1)
    axs[0].set_title(title,size=10, color="white")
    axs[0].patch.set_edgecolor('white')
    axs[0].patch.set_linewidth('2')
    axs[0].text(stampsize//2, -stampsize//8, r'1 arcmin$^{2}$', horizontalalignment='center', verticalalignment='center', color="white", animated=True)

    
    #plt.colorbar(im,cax=cax)

    comp=1
    xlabel=r"$g_{%i}$"%(comp)
    ylabel=r"$\langle \hat{g_{%i}} - g_{%i} \rangle$"%(comp,comp)
    color_plot_ax(axs[1],x,y,z=None,colorlog=False,xtitle=xlabel,ytitle=ylabel,ctitle="",ftsize=16,xlim=[-0.1,0.1], ylim=[-0.5,0.5],cmap=None,filename=None, colorbar=False,linreg=True)
    axs[1].set_title(title,size=10, color="white")
    
    #ax.axis('off')
    fig.tight_layout()
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gift
    plt.close(fig)

def color_plot_ax(ax,x,y,z=None,colorlog=True,xtitle="",ytitle="",ctitle="",ftsize=16,xlim=None, ylim=None,cmap=None,filename=None, colorbar=True,linreg=True):

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
        ax.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='$\mu_{1}$: %.4f $\pm$ %.4f \n  c$_{1}$: %.4f $\pm$ %.4f'%(m,merr,c, cerr ))
        ax.legend(loc='upper left', prop={'size': ftsize-6}, labelcolor="white")
        
    if colorlog: 
        c=abs(c)
        colornorm=LogNorm( vmin=np.nanmin(c), vmax=np.nanmax(c))
    else: colornorm=None


    sct=ax.scatter(x, y,c=z, norm=colornorm, marker=".",alpha=0.7,cmap=cmap)
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
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('black')

def plot_neighbors():
    
    psf_image = galsim.ImageF(stampsize, stampsize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    #fmin=10
    #fmax=70
    #df=2*(fmax-fmin)/NFRAMES
    #density_list=np.concatenate([[fmin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()

    #density_list=[10,10,10,20,20,20,30,30,30,40,40,40, 50, 50, 50, 60, 60, 60, 70, 70, 70]
    density_list=[10,20,30,40,50,60,70,80]
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
                gal_image = galsim.ImageF(stampsize, stampsize )
                poslist=[]
                for j in range(int(d)):
                    x=np.random.uniform(0,stampsize)
                    y=np.random.uniform(0,stampsize)
                    pos = galsim.PositionD(x,y)
                    poslist.append(pos)
            
                    gal = galsim.Gaussian(sigma=.5, flux=1.e2)
                    gal = gal.shear(g1=s, g2=0.5)
                    gal = gal.shear(g1=s, g2=0.0)
                    psf = galsim.Gaussian(sigma=3.55*psfpixelscale, flux=1.0)
                    psf.drawImage(psf_image, scale=scale)
                    #psf = psf.shear(g1=0.2, g2=0.1)
                    galconv = galsim.Convolve([gal,psf])
                    galconv.drawImage(gal_image, center=pos, scale=scale, add_to_image=True)

                obse=[]
                for pos in poslist:
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    
                    try:
                        res = galsim.hsm.FindAdaptiveMom(gal_image,  weight=None, hsmparams=hsmparams, guess_centroid=pos)
                        ada_g1 = res.observed_shape.g1
                    except:
                        continue
                    '''
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    try:
                        result = galsim.hsm.EstimateShear(gal_image, psf_image,  hsmparams=hsmparams, guess_centroid=pos,  shear_est="KSB")
                        ada_g1=result.corrected_g1
                    except:
                        continue
                    '''
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

def main():
    plot_neighbors()

if __name__ == "__main__":
    main()
