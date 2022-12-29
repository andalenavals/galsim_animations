
AUXNAME="deleteme.png"
NFRAMES=100
FIGZISE=(8,3)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PSF biases')
    parser.add_argument('--workdir',
                        default='out', 
                        help='diractory of work')   
    args = parser.parse_args()

    return args

def make_plot(galimg, psfimg, galpsfimg, plotname='test.png'):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np
    import pandas as pd

    fig, axs= plt.subplots(1, 3, figsize=(8,3))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax0=axs[0]; ax1=axs[1]; ax2=axs[2]
    arrays=[img.array for img in [galimg, psfimg, galpsfimg]]
    cmap='gray'
    norm=None
    for ax, dat,title in zip([ax0,ax1,ax2],arrays,["Galaxy","PSF", "Observed galaxy"]):
        vmin=0; vmax=0.1
        #vmin=None; vmax=None
        ax.imshow(dat,cmap=cmap,norm=norm, vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 63)
        ax.set_title(title,size=10, color="white")
        ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.savefig(plotname, transparent=False) #transparent does not work with gift

def plots_psf_size():
    import galsim
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as anime
    
    #make images
    stampsize=64
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

    
    # Make galaxy
    gal = galsim.Gaussian(sigma=2.0, flux=3.0)
    gal = gal.shear(g1=0.5, g2=0.0)
    image= gal.drawImage(gal_image)

   
    fig, ax = plt.subplots(figsize=(8,3))
    ims=[]

    d=np.pi/NFRAMES
    sigmalist=[2.0+7.0*np.sin(d*i) for i in range(NFRAMES)]
    for sigma in sigmalist:
        # Make PSF
        psf = galsim.Gaussian(sigma=sigma, flux=1.0)
        #psf = psf.shear(g1=0.2, g2=0.1)
        image= psf.drawImage(psf_image)

        ##CONVOLVED
        galconv = galsim.Convolve([gal,psf])
        image=galconv.drawImage(galpsf_conv_image)

        make_plot(gal_image, psf_image, galpsf_conv_image, plotname=AUXNAME)
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ani = anime.ArtistAnimation(fig, ims, interval=60, blit=True)
    ani.save('psf_size.gif')
    plt.close(fig)

def plots_psf_anisotropy():
    import galsim
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as anime
    import galsim_hub
    from astropy.table import Table
    catalog = Table([[5., 3. ,2.], [29., 20., 15.], [0.0, 0.2, 0.5] ],
             names=['flux_radius', 'mag_auto', 'zphot'])
    
    #make images
    stampsize=64
    gal_image = galsim.ImageF(stampsize, stampsize )
    psf_image = galsim.ImageF(stampsize, stampsize )
    galpsf_conv_image = galsim.ImageF(stampsize, stampsize )

    
    # Make galaxy
    model = galsim_hub.GenerativeGalaxyModel('https://zenodo.org/record/7457343/files/model.tar.gz')
    profiles = model.sample(catalog)
    gal=profiles[0]
    gal.drawImage(gal_image)
    #gal = galsim.Gaussian(sigma=5.0, flux=2.5)
    #gal = gal.shear(g1=0.5, g2=0.0)
    #image= gal.drawImage(gal_image)

   
    fig, ax = plt.subplots(figsize=(8,3))
    ims=[]
    
    n1=NFRAMES//2
    n2=NFRAMES-n1

    d=2*np.pi/n1
    g1list=[0.8*np.sin(i*d) for i in range(n1)]
    g2list=[0.0]*n1
    
    g2list+=g1list
    g1list+=[0.0]*n1
    
    for g1,g2 in zip(g1list,g2list):
        # Make PSF
        psf = galsim.Gaussian(sigma=1.5, flux=1.0)
        psf = psf.shear(g1=g1, g2=g2)
        psf.drawImage(psf_image)

        ##CONVOLVED
        galconv = galsim.Convolve([gal,psf])
        galconv.drawImage(galpsf_conv_image)

        make_plot(gal_image, psf_image, galpsf_conv_image, plotname=AUXNAME)
        im = ax.imshow(plt.imread(AUXNAME), animated = True)
        ims.append([im])

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ani = anime.ArtistAnimation(fig, ims, interval=60, blit=True)
    ani.save('psf_anisotropy.gif')
    plt.close(fig)
   
def main():
    
    #plots_psf_size()
    plots_psf_anisotropy()

    

if __name__ == "__main__":
    main()
