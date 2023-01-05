AUXNAME="deleteme2.png"
NFRAMES=100
NREP=10
FIGZISE=(8,3)
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

def make_plot(galimg, galimgnoise,plotname='test.png', vmin=None, vmax=None, title=""):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    import pandas as pd
    
    fig, axs= plt.subplots(1, 2, figsize=FIGZISE)
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax0=axs[0]; ax1=axs[1]
    arrays=[img.array for img in [galimg, galimgnoise]]
    cmap='gray'
    
    #vmin=None; vmax=None
    #for ax, dat,title,vmin,vmax in zip([ax0,ax1],arrays,["Galaxy","Poisson Noise"],[vmin,None],[vmax,None]):
    for ax, dat,tit in zip([ax0,ax1],arrays,["Galaxy",title]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #im=ax.imshow(dat,cmap=cmap,norm=LogNorm( vmin=vmin, vmax=vmax))
        im=ax.imshow(dat,cmap=cmap,norm=None, vmin=vmin, vmax=vmax)
        ax.set_ylim(0,stampsize-1)
        ax.set_title(tit,size=10, color="white")
        ax.patch.set_edgecolor('white')
        ax.patch.set_linewidth('2')
        plt.colorbar(im,cax=cax)
        #ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gift
    plt.close(fig)

def plots_poisson_noise():
    import galsim
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as anime
    import astropy.visualization
    
    #make images
    gal_image = galsim.ImageF(stampsize, stampsize )
    gpoisson_image = galsim.ImageF(stampsize, stampsize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    #flux_list=[100+2.3**i for i in range(int(NFRAMES//2))]
    fmin=1.e3
    fmax=1.e7
    df=2*(fmax-fmin)/NFRAMES
    flux_list=np.concatenate([[fmin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()
    flux_list+=np.flip(flux_list).tolist()

    vmin=0
    vmax=2000
    
    vis=astropy.visualization.ZScaleInterval(nsamples=1000, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5, max_iterations=5)
    
    rng = galsim.BaseDeviate(123)
    #rng = None
    for i in range(len(flux_list)):
        gal = galsim.Gaussian(sigma=5.0, flux=flux_list[i])
        gal = gal.shear(g1=0.5, g2=0.0)
        psf = galsim.Gaussian(sigma=2.5, flux=1.0)
        psf = psf.shear(g1=0.2, g2=0.1)
        galconv = galsim.Convolve([gal,psf])
        galconv.drawImage(gal_image, scale=scale)
        galconv.drawImage(gpoisson_image, scale=scale)
        
        poisson_noise = galsim.PoissonNoise(rng, sky_level=0.)
        gpoisson_image.addNoise(poisson_noise)

        make_plot(gal_image, gpoisson_image, plotname=AUXNAME, vmin=vmin, vmax=vmax, title="Poisson noise")
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
    import galsim
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as anime
    import astropy.visualization
    
    #make images
    gal_image = galsim.ImageF(stampsize, stampsize )
    ggaussian_image = galsim.ImageF(stampsize, stampsize )

    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]
    
    # Make galaxy

    smin=1.e0
    smax=1.e5
    df=2*(smax-smin)/NFRAMES
    sigmalist=np.concatenate([[smin+i*df]*NREP for i in range(int(NFRAMES//(2*NREP)))]).tolist()
    sigmalist+=np.flip(sigmalist).tolist()

    vmin=0
    vmax=3.e4
    
    rng = galsim.BaseDeviate(123)
    #rng = None
    for i in range(len(sigmalist)):
        gal = galsim.Gaussian(sigma=5.0, flux=1.e7)
        gal = gal.shear(g1=0.5, g2=0.0)
        psf = galsim.Gaussian(sigma=2.5, flux=1.0)
        psf = psf.shear(g1=0.2, g2=0.1)
        galconv = galsim.Convolve([gal,psf])
        galconv.drawImage(gal_image, scale=scale)
        galconv.drawImage(ggaussian_image, scale=scale)
        
        gaussian_noise = galsim.GaussianNoise(rng, sigma=sigmalist[i])
        ggaussian_image.addNoise(gaussian_noise)

        make_plot(gal_image, ggaussian_image, plotname=AUXNAME, vmin=vmin, vmax=vmax, title="Gaussian read noise")
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
    plots_gaussian_noise()

if __name__ == "__main__":
    main()
