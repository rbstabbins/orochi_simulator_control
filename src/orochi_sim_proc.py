"""A library of functions for processing OROCHI Simulator Images.

Roger Stabbins
Rikkyo University
21/04/2023
"""
from pathlib import Path
import os
import astropy.io.fits as fitsio
from collections import OrderedDict
import cv2
# import dict2xml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib as mpl
from matplotlib import ticker as mticker
import numpy as np
import pandas as pd
from roipoly import RoiPoly
import scipy.optimize as opt
import scipy.signal as sig
import scipy.ndimage as ndi
from shutil import copytree, copy
import tifffile as tiff
from typing import Tuple, Dict, Union, List
import orochi_sim_ctrl as osc

FIG_W = 10 # figure width in inches

# set default Window/ROI Size information
WIN_S = 100 # size of window in pixels
Y_TRIM = 20 #100 # fine adjustment of the window centre
X_TRIM = 100 #20 # fine adjustment of the window centre
CNTR = [(1200//2)-(WIN_S//2)+Y_TRIM, (1920//2)-(WIN_S//2)+X_TRIM] # centre of the window
OFFSET = 275 # offset in pixels for a 10 cm camera displacement at 80 cm object distance
WINDOWS = { # window coordinartes for each camera
    0: [ CNTR[0], CNTR[1]-OFFSET, WIN_S, WIN_S],
    1: [ CNTR[0]-OFFSET, CNTR[1]-OFFSET, WIN_S, WIN_S],
    2: [ CNTR[0]+OFFSET, CNTR[1]+OFFSET, WIN_S, WIN_S],
    3: [ CNTR[0]-OFFSET, CNTR[1], WIN_S, WIN_S],
    4: [ CNTR[0], CNTR[1]+OFFSET, WIN_S, WIN_S],
    5: [ CNTR[0]+OFFSET, CNTR[1], WIN_S, WIN_S],
    6: [ CNTR[0]+OFFSET, CNTR[1]-OFFSET, WIN_S, WIN_S],
    7: [ CNTR[0]-OFFSET, CNTR[1]+OFFSET, WIN_S, WIN_S]
}

# incidence (°), cam x-displacement (cm), cam y-displacement (cm), emission (°), azimuth (°), phase (°)

CAM_ANGLES = { 
    0: [30.00, 0.10, 0.00, 7.125, 180.000, 37.125],
    1: [30.00, 0.10, -0.10, 10.025, 225.000, 37.697],
    2: [30.00, -0.10, 0.10, 10.025, 45.000, 23.887],
    3: [30.00, 0.00, -0.10, 7.125, 270.000, 30.758],
    4: [30.00, -0.10, 0.00, 7.125, 0.000, 22.875],
    5: [30.00, 0.00, 0.10, 7.125, 90.000, 30.758],
    6: [30.00, 0.10, 0.10, 10.025, 135.000, 37.697],
    7: [30.00, -0.10, -0.10, 10.025, 315.000, 23.887]
}

class Image:
    """Super class for handling image import and export.
    """
    def __init__(self, scene_path_in: Path, scene_path_out: Path, channel: str, img_type: str, roi: bool=False) -> None:
        """Initialise properties. Most of these are populated during image load.
        """
        self.scene_dir = Path(scene_path_in, channel) # input directory
        self.products_dir = scene_path_out
        self.scene = scene_path_in.name
        self.channel = channel
        self.img_type = img_type
        self.camera = None
        self.emission_angle = None
        self.azimuth_angle = None
        self.phase_angle = None
        self.serial = None
        self.width = None
        self.height = None
        self.cwl = None
        self.fwhm = None
        self.fnumber = None
        self.flength = None
        self.exposure = None
        self.roix = None
        self.roiy = None
        self.roiw = None
        self.roih = None
        self.winx = None
        self.winy = None
        self.winw = None
        self.winh = None
        self.roi = roi
        self.polyroi = None
        self.units = ''
        self.n_imgs = None
        self.img_one = None
        self.img_ave = None
        self.img_std = None
        self.img_err = None 
        self.dif_img = None   

    def image_display(self,
                      statistic: str='averaged',
                      ax: object=None, histo_ax: object=None,
                      threshold: float=None,
                      draw_roi: bool=False, polyroi: bool=False,
                      window: Union[bool, str]=False,
                      context: object=None,
                      vmin: float=None, vmax: float=None) -> None:
        """Display the image mean and standard deviation in one frame.

        :param ax: image axis object, defaults to None
        :type ax: object, optional
        :param histo_ax: histogram axis object, defaults to None
        :type histo_ax: object, optional
        :param noise: use the noise image, defaults to False
        :type noise: bool, optional
        :param snr: use the SNR image, defaults to False
        :type snr: bool, optional
        :param threshold: set all pixels below this value to NaN, defaults to None
        :type threshold: float, optional
        :param draw_roi: draw the current ROI, defaults to False
        :type roi: bool, optional
        :param polyroi: set all values outside polyROI to np.nan, defaults to False
        :type polyroi: bool, optional
        :param window: use the default window size, and if in conjunction with
            ROI and polyroi, draw PolyROI and ROI on the image, defaults to False
        :type window: bool, optional
        :param context: context Image object, defaults to None
        :type context: Image object, optional
        :param vmin: minimum value for the colorbar, defaults to None
        :type vmin: float, optional
        :param vmax: maximum value for the colorbar, defaults to None
        :type vmax: float, optional
        """
        # set the size of the window
        if ax is None:
            fig, axs = plt.subplots(2,1,figsize=(FIG_W/2, FIG_W))
            ax = axs[0]
            histo_ax = axs[1]
            fig.suptitle(f'scene: {self.scene} ({self.img_type})')

        if statistic=='single-frame':
            img = self.img_one
            label = self.units+' Single Frame'
        elif statistic=='averaged':
            img = self.img_ave
            label = self.units+' Averaged'
        elif statistic=='single-frame-noise':
            img = self.img_std
            label = self.units+' Single Frame Noise'
        elif statistic=='averaged-noise':
            img = self.img_err
            label = self.units+' Averaged Noise'
        elif statistic=='single-frame-snr':
            out = np.full(self.img_one.shape, 0.0)
            np.divide(np.abs(self.img_one), self.img_std, out=out, where=self.img_one!=0)
            img = out
            label = self.units+' Single Frame SNR'
        elif statistic=='averaged-snr':
            out = np.full(self.img_ave.shape, 0.0)
            np.divide(np.abs(self.img_ave), self.img_err, out=out, where=self.img_ave!=0)
            img = out
            label = self.units+' Averaged SNR'
        elif statistic=='dif_img':
            img = self.dif_img
            label = 'Median Normalised Difference'
        else:
            img = self.img_ave
            label = self.units

        if threshold:
            img = np.where(img < threshold, np.nan, img)

        if window is True:
            # set image coordinate limits
            win_y = self.winy
            win_x = self.winx
            win_h = self.winh
            win_w = self.winw
        elif window == 'roi':
            # set image coordinate limits
            win_y = self.roiy
            win_x = self.roix
            win_h = self.roih
            win_w = self.roiw
        elif window == 'roi_centred':
            # set image coordinate limits
            self.winy = (self.roiy + self.roih//2) - self.winh//2
            self.winx = (self.roix + self.roiw//2) - self.winw//2
            win_y = self.winy
            win_x = self.winx
            win_h = self.winh
            win_w = self.winw
        else:
            win_y = 0
            win_x = 0
            win_h = self.height
            win_w = self.width

        if polyroi:
            img = img.copy()
            img = np.where(self.polyroi, img, np.nan)

        win_img = img[win_y:win_y+win_h, win_x:win_x+win_w]
        extent = [win_x, win_x+win_w, win_y+win_h, win_y] # coordinates of (left, right, bottom, top)

        col = channel_cols(self.camera)     
        if histo_ax is not None:
            # add histogram      
            if statistic=='dif_img':
                img = self.dif_img
                title = f'Difference Histogram'
            elif draw_roi:
                img = img[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]
                title = f'ROI Histogram'            
            elif window:
                img = win_img
                title = f'Window Histogram'
            else:
                img = win_img
                title = f'Image Histogram'
                vmax = np.nanmax(img)
            counts, bins = np.histogram(img[np.nonzero(np.isfinite(img))], bins=128)  
            
            # set range according to median and std of ROI
            roi_ave = np.nanmean(img)
            roi_std = np.nanstd(img)

            if roi_std == 0:
                roi_std = roi_ave/10

            # TODO still some problems occuring here            
            if vmin is None:
                vmin = roi_ave - (roi_std * 10)                
                vmin = np.clip(vmin, np.nanmin(win_img), np.nanmax(win_img))  
                if not np.isfinite(vmin):
                    print('error')         
            if vmax is None:
                vmax = roi_ave + (roi_std * 5)
                vmax = np.clip(vmax, np.nanmin(win_img), np.nanmax(win_img))
                if not np.isfinite(vmax):
                    print('error')
            x_2_sigma = lambda x: (x - roi_ave) / roi_std
            sigma_2_x = lambda x: (x * roi_std) + roi_ave
            
            histo_ax.hist(bins[:-1], bins, weights=counts,
                        label=f'{int(self.cwl)} nm ({self.camera})',
                        color=col,
                        log=True, fill=False, stacked=True, histtype='step')
            histo_ax.set_xlabel(label)
            # histo_ax.set_box_aspect(im_ratio)
            histo_ax.legend()
            histo_ax.set_title(title)

        if statistic=='dif_img':
            cmap = 'seismic'
            vmax = 1.0 # np.max(np.abs(win_img))
            vmin = -1.0 # -vmax
        else:
            cmap = 'viridis'

        ave = ax.imshow(win_img, origin='upper', vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)

        # draw window/ROI
        if draw_roi:
            rect = patches.Rectangle((self.roix, self.roiy), self.roiw, self.roih, linewidth=1, edgecolor='r', facecolor='none')        
            ax.add_patch(rect)        

        im_ratio = win_img.shape[0] / win_img.shape[1]

        cbar = plt.colorbar(ave, ax=ax, fraction=0.047*im_ratio, pad=0.08)           
        # add axis to the color bar centered on the mean and extending in std. dev.
        cbar2 = cbar.ax.secondary_yaxis('left',functions=(x_2_sigma, sigma_2_x))

        ax.set_title(f'Device {self.camera} ({int(self.cwl)} nm)')

        if context is not None:
            # draw context image
            # normalise to maximum of the reflectance win_img          
            context_img = context.img_ave
            context_img = context_img[win_y:win_y+win_h, win_x:win_x+win_w]
            context_img = context_img * np.nanmax(win_img) / np.nanmax(context_img)
            ax.imshow(context_img, origin='upper', cmap='gray', alpha=0.5, extent=extent)                        

        if ax is None:
            plt.tight_layout()
            plt.show()

        plt.draw()
        yticks = cbar2.get_yticks()
        yticks = yticks[yticks % 1 == 0] # drop non-integer entries
        new_yticks = [fr"{t:.0f}" if t != 0 else r"$\mu$" for t in yticks]
        cbar2.yaxis.set_major_locator(mticker.FixedLocator(yticks.tolist()))
        cbar2.set_yticklabels(new_yticks)                
        cbar.set_label(r"Hist. $\sigma$ | Val.", y=1.1, rotation=0, labelpad=-15)        
        
        return ax

    def save_image(self, 
                   float32: bool=True,
                   uint8: bool=True,
                   uint16: bool=True,
                   fits: bool=True,
                   roi: bool=False) -> None:
        """Save the image and error image to the formats indicated with the flags.

        :param float32: output TIFF file in Float 64, defaults to True
        :type float32: bool, optional
        :param uint8: output TIFF file in UINT8, defaults to False
        :type uint8: bool, optional
        :param uint16: output TIFF file in UINT16, defaults to False
        :type uint16: bool, optional
        :param fits: output FITS file in Float 64, defaults to False
        :type fits: bool, optional
        """    
        print(f'Saving {self.channel} {self.scene} images to', end='')    
        if float32:            
            # write image with float 64 method            
            output = 'float32'
            self.save_tiff(output, roi)
            print(' TIFF float32,', end='')

        if uint8:
            # write image with uint8 method
            output = 'uint8'
            self.save_tiff(output, roi)
            print(' TIFF uint8,', end='')

        if uint16:
            # write image with uint16 method                        
            output = 'uint16'
            self.save_tiff(output, roi)
            print(' TIFF uint16,', end='')

        if fits:
            # write image with fits method
            self.save_fits(roi)
            print(' fits,', end='')
        
        print(' formats.')

    def save_tiff(self, dtype: str, roi: bool=False) -> None:
        """Save the image and error image to TIFF file

        :param dtype: output datatype, one of 'float32', 'uint8', 'uint16'
        :type dtype: str        
        """

        if roi:
            img_one = self.img_one[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_ave = self.img_ave[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_std = self.img_std[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_err = self.img_err[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()

        else:
            img_one = self.img_one.copy()
            img_ave = self.img_ave.copy()
            img_std = self.img_std.copy()
            img_err = self.img_err.copy()

        metadata={
                    'scene': self.scene,
                    'image-type': self.img_type,
                    'camera': self.camera,
                    'phase': self.phase_angle,
                    'serial': self.serial,
                    'cwl': self.cwl,
                    'fwhm': self.fwhm,
                    'f-number': self.fnumber,
                    'f-length': self.flength,
                    'exposure': self.exposure,
                    'units': self.units,
                    'n_imgs': self.n_imgs
                }

        if dtype == 'float32':
            img_one = img_one.astype(np.float32)
            img_ave = img_ave.astype(np.float32)
            img_std = img_std.astype(np.float32)
            img_err = img_err.astype(np.float32)
        elif dtype == 'uint8':
            if self.img_type == 'rfl':
                img_one = np.floor(200*img_one).astype(np.uint8)
                img_ave = np.floor(200*img_ave).astype(np.uint8)
                img_std = np.floor(200*img_std).astype(np.uint8)
                img_err = np.floor(200*img_err).astype(np.uint8)
                metadata['units'] = f'{self.units} x200'
            else:
                img_one = np.clip(np.floor(img_one/16), 0, 255).astype(np.uint8)
                img_ave = np.clip(np.floor(img_ave/16), 0, 255).astype(np.uint8)
                img_std = np.clip(np.floor(img_std/16), 0, 255).astype(np.uint8)
                img_err = np.clip(np.floor(img_err/16), 0, 255).astype(np.uint8)
        elif dtype == 'uint16':
            if self.img_type == 'rfl':
                img_one = np.floor(10000*img_one).astype(np.uint16)
                img_ave = np.floor(10000*img_ave).astype(np.uint16)
                img_std = np.floor(10000*img_std).astype(np.uint16)
                img_err = np.floor(10000*img_err).astype(np.uint16)
                metadata['units'] = f'{self.units} x10000'
            else:
                img_one = np.floor(img_one).astype(np.uint16)
                img_ave = np.floor(img_ave).astype(np.uint16)
                img_std = np.floor(img_std).astype(np.uint16)
                img_err = np.floor(img_err).astype(np.uint16)
        
        cwl_str = str(int(self.cwl))
        cam_num = str(self.camera) 
        filename = cam_num+'_'+cwl_str+'_'+self.img_type

        product_dir = Path(self.products_dir, self.img_type)
        product_dir.mkdir(parents=True, exist_ok=True)        

        out_dir = Path(product_dir, dtype)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        one_dir = Path(out_dir, 'one')
        ave_dir = Path(out_dir, 'ave')
        std_dir = Path(out_dir, 'std')
        err_dir = Path(out_dir, 'err')
        one_dir.mkdir(parents=True, exist_ok=True)
        ave_dir.mkdir(parents=True, exist_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)
        err_dir.mkdir(parents=True, exist_ok=True)

        img_one_file =str(Path(one_dir, filename+'_one').with_suffix('.tif'))
        img_ave_file =str(Path(ave_dir, filename+'_ave').with_suffix('.tif'))
        img_std_file =str(Path(std_dir, filename+'_std').with_suffix('.tif'))
        img_err_file =str(Path(err_dir, filename+'_err').with_suffix('.tif'))

        # write camera properties to TIF using ImageJ metadata 
        tiff.imwrite(img_one_file, img_one, imagej=True, metadata=metadata)       
        tiff.imwrite(img_ave_file, img_ave, imagej=True, metadata=metadata)
        tiff.imwrite(img_std_file, img_std, imagej=True, metadata=metadata)
        tiff.imwrite(img_err_file, img_err, imagej=True, metadata=metadata)

    def save_fits(self, roi: bool=False) -> None:
        """Save the image and error image to FITS files
        """                
        if roi:
            img_one = self.img_one[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_ave = self.img_ave[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_std = self.img_std[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()
            img_err = self.img_err[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw].copy()

        else:
            img_one = self.img_one.copy()
            img_ave = self.img_ave.copy()
            img_std = self.img_std.copy()
            img_err = self.img_err.copy()

        cwl_str = str(int(self.cwl))
        cam_num = str(self.camera) 
        filename = cam_num+'_'+cwl_str+'_'+self.img_type

        product_dir = Path(self.products_dir, self.img_type)
        product_dir.mkdir(parents=True, exist_ok=True)

        out_dir = Path(product_dir, 'fits')
        out_dir.mkdir(parents=True, exist_ok=True)
        
        one_dir = Path(out_dir, 'one')
        ave_dir = Path(out_dir, 'ave')
        std_dir = Path(out_dir, 'std')
        err_dir = Path(out_dir, 'err')

        one_dir.mkdir(parents=True, exist_ok=True)
        ave_dir.mkdir(parents=True, exist_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)
        err_dir.mkdir(parents=True, exist_ok=True)

        img_one_file =str(Path(one_dir, filename+'_one').with_suffix('.fits'))
        img_ave_file =str(Path(ave_dir, filename+'_ave').with_suffix('.fits'))
        img_std_file =str(Path(std_dir, filename+'_std').with_suffix('.fits'))
        img_err_file =str(Path(err_dir, filename+'_err').with_suffix('.fits'))
        
        hdu = fitsio.PrimaryHDU(img_one)
        hdu.header['scene'] = self.scene
        hdu.header['type'] = self.img_type
        hdu.header['camera'] = int(self.camera)
        hdu.header['serial'] = int(self.serial)
        hdu.header['cwl'] = self.cwl
        hdu.header['fwhm'] = self.fwhm
        hdu.header['f-number'] = self.fnumber
        hdu.header['f-length'] = self.flength
        hdu.header['exposure'] = self.exposure
        hdu.header['units'] = self.units
        hdu.writeto(img_one_file, overwrite=True)
        
        hdu = fitsio.PrimaryHDU(img_ave)
        hdu.header['scene'] = self.scene
        hdu.header['type'] = self.img_type
        hdu.header['camera'] = int(self.camera)
        hdu.header['serial'] = int(self.serial)
        hdu.header['cwl'] = self.cwl
        hdu.header['fwhm'] = self.fwhm
        hdu.header['f-number'] = self.fnumber
        hdu.header['f-length'] = self.flength
        hdu.header['exposure'] = self.exposure
        hdu.header['units'] = self.units
        hdu.writeto(img_ave_file, overwrite=True)

        hdu = fitsio.PrimaryHDU(img_std)
        hdu.header['scene'] = self.scene
        hdu.header['type'] = self.img_type
        hdu.header['camera'] = int(self.camera)
        hdu.header['serial'] = int(self.serial)
        hdu.header['cwl'] = self.cwl
        hdu.header['fwhm'] = self.fwhm
        hdu.header['f-number'] = self.fnumber
        hdu.header['f-length'] = self.flength
        hdu.header['exposure'] = self.exposure
        hdu.header['units'] = self.units
        hdu.writeto(img_std_file, overwrite=True)

        hdu = fitsio.PrimaryHDU(img_err)
        hdu.header['scene'] = self.scene
        hdu.header['type'] = self.img_type
        hdu.header['camera'] = int(self.camera)
        hdu.header['serial'] = int(self.serial)
        hdu.header['cwl'] = self.cwl
        hdu.header['fwhm'] = self.fwhm
        hdu.header['f-number'] = self.fnumber
        hdu.header['f-length'] = self.flength
        hdu.header['exposure'] = self.exposure
        hdu.header['units'] = self.units
        hdu.writeto(img_err_file, overwrite=True)

    def image_load(self, n_imgs: int=None, mode: str='mean', stack: np.ndarray=None) -> None:
        """Load images from the scene directory for the given type,
        populate properties, and compute averages and standard deviation.
        """

        # get list of images of given type in the scene directory
        files = list(self.scene_dir.glob('*'+self.img_type+'.tif'))
        if files == []:
            raise FileNotFoundError(f'Error: no {self.img_type} images found in {self.scene_dir}')
        
        # set n_imgs
        if n_imgs == None:
            self.n_imgs = len(files)
        else:
            self.n_imgs = n_imgs

        self.units = 'Raw DN'

        img_list = []
        
        for f, file in enumerate(files[:self.n_imgs]):
            try:
                img = tiff.TiffFile(file)
            except ValueError:
                print('bad file')
            img_arr = img.asarray()
            img_list.append(img_arr)
            meta = img.imagej_metadata
            self.camera = self.check_property(self.camera, meta['camera'])
            self.serial = self.check_property(self.serial, meta['serial'])
            self.cwl = self.check_property(self.cwl, meta['cwl'])

            # NOTE - error for 650 nm filter FWHM - recorded as 50 nm, actually only 10 nm
            if self.cwl == 650:
                self.fwhm = 10
                meta['fwhm'] = 10                
            
            self.fwhm = self.check_property(self.fwhm, meta['fwhm'])

            self.fnumber = self.check_property(self.fnumber, meta['f-number'])
            self.flength = self.check_property(self.flength, meta['f-length'])
            self.exposure = self.check_property(self.exposure, meta['exposure'])
            # try:
            #     self.roiy = self.check_property(self.roiy, meta['roiy'])
            #     self.roix = self.check_property(self.roix, meta['roix'])
            #     self.roih = self.check_property(self.roih, meta['roih'])
            #     self.roiw = self.check_property(self.roiw, meta['roiw'])
            # except (KeyError, ValueError):
            # read the camera_config file to get the ROI

            # check the dimensions, and check the x and y properties for inversion

            self.height = self.check_property(self.height, img_arr.shape[0])
            self.width = self.check_property(self.width, img_arr.shape[1])
            camera_info = osc.load_camera_config(Path(self.scene_dir, '..').resolve())
            cam_name = f'DMK 33GX249 {int(self.serial)}'
            cam_props = camera_info[cam_name]
            # currently roiy and roix labels are inverted - so correct on load in here
            self.roiy = cam_props['roix']
            self.roix = cam_props['roiy']
            self.roih = cam_props['roiw']
            self.roiw = cam_props['roih']
            self.winy = WINDOWS[self.camera][0]
            self.winx = WINDOWS[self.camera][1]
            self.winh = WINDOWS[self.camera][2]
            self.winw = WINDOWS[self.camera][3]

            # get phase angle information from camera number look up
            angle_info = CAM_ANGLES[self.camera]
            self.emission_angle = angle_info[3]
            self.azimuth_angle = angle_info[4]
            self.phase_angle = angle_info[5]

        img_stk = np.dstack(img_list)
        
        # single image
        self.img_one = img_stk[:,:,0].astype(np.float64)

        # average image
        if mode == 'mean':
            self.img_ave = np.mean(img_stk, axis=2)
        elif mode == 'median':
            self.img_ave = np.median(img_stk, axis=2)
        
        # standard deviation image
        if self.n_imgs > 1:
            self.img_std = np.std(img_stk, axis=2)
        else:
            # estimate the standard deviation as the shot noise on the image
            k_adc = 4.7 # e-/DN
            self.img_std = np.sqrt(self.img_ave / k_adc)

        # standard error image
        self.img_err = self.img_std / np.sqrt(self.n_imgs)

        print(f'Loaded {f+1} images ({self.img_type}) for: {self.camera} ({int(self.cwl)} nm)')
        if stack is not None:
            return img_stk, files

    def roi_image(self, polyroi: bool=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Returns the region of interest of the image.
        """
        if polyroi:
            img_one = np.where(self.polyroi, self.img_one, np.nan)
            img_ave = np.where(self.polyroi, self.img_ave, np.nan)
            img_std = np.where(self.polyroi, self.img_std, np.nan)
            img_err = np.where(self.polyroi, self.img_err, np.nan)
        else:
            img_one = self.img_one
            img_ave = self.img_ave
            img_std = self.img_std
            img_err = self.img_err

        img_one = img_one[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]
        img_ave = img_ave[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]
        img_std = img_std[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]
        img_err = img_err[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]

        return img_one, img_ave, img_std, img_err

    def roi_std(self, polyroi: bool=False) -> np.ndarray:
        """Returns the region of interest of the error image.
        """
        if polyroi:
            img = np.where(self.polyroi, self.img_std, np.nan)
        else:
            img = self.img_std

        roi_img = img[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]

        return roi_img

    def check_property(self, obj_prop, metadata):
        """Check that the metadata for a given image property is consistent"""
        obj_set = obj_prop != None
        if isinstance(obj_prop, float):
            obj_meta_match = np.allclose(obj_prop, metadata, equal_nan=True)
        else:
            obj_meta_match = obj_prop == metadata
        if obj_set and not obj_meta_match:
            raise ValueError(f'Error: image metadata anomaly - {obj_prop} != {metadata}')
        else:
            return metadata

    def correct_exposure(self) -> None:
        """Apply exposure correction to all images, and update units
        """
        if self.units == 'DN/s':
            print('Exposure already corrected.')
            return

        self.img_one = self.img_one / self.exposure
        self.img_ave = self.img_ave / self.exposure
        self.img_std = self.img_std / self.exposure # assume exposure err. negl.
        self.img_err = self.img_err / self.exposure
        self.units = 'DN/s'

    def set_roi(self, 
                threshold: int=None, 
                roi_size: Union[int, Tuple[int, int]]=None, 
                roi_params: Tuple=None,
                cross_hair_is_centre: bool=False) -> None:
        """Set a rectangular region of interest
        
        :param threshold: set all pixels below this value to NaN, defaults to None
        :type threshold: int, optional
        :param roi_size: size of the ROI, either square size or [h, w], defaults to None
        :type roi_size: Union[int, Tuple[int, int]], optional
        :param roi_params: parameters of the ROI, defaults to None
        :type roi_params: Tuple, optional
        :param cross_hair_is_centre: if True, the cross hair is the centre of the ROI, defaults to False
        :type cross_hair_is_centre: bool, optional
        :return: list of the ROI parameters: [y, x, h, w]
        :rtype: List
        """

        if self.roi:
            _, img, _, _ = self.roi_image()
        else:
            img = self.img_ave    

        # if not 8 bit convert for display
        if img.dtype != np.uint8:
            # _, img_ave, _, _ = self.roi_image()
            # if img_ave.shape == (0,0):
            img_ave = self.img_ave
            img = np.clip(np.floor(img * 255/np.nanmax(img_ave)), 0, 255).astype(np.uint8)

        if roi_params is None:
            title = f'ROI Selection: {self.camera}_{self.cwl}'            
            roi = cv2.selectROI(title, img, showCrosshair=True) # roi output is (x,y,w,h)
            # switch order of roi
            roi = (roi[1], roi[0], roi[3], roi[2])           
            cv2.destroyWindow(title)   

            if roi == (0,0,0,0):
                print('ROI not set - trying again')
                roi = cv2.selectROI(title, img, showCrosshair=True) # roi output is (x,y,w,h)
                # switch order of roi
                roi = (roi[1], roi[0], roi[3], roi[2])           
                cv2.destroyWindow(title)  
            
            if roi == (0,0,0,0):
                print('ROI not set - resorting to previous ROI')
                return [self.roiy, self.roix, self.roih, self.roiw]
            
        else:
            roi = roi_params

        if roi_size is not None:
            if isinstance(roi_size, int) :
                self.roih = roi_size
                self.roiw = roi_size
            else:
                self.roih = roi_size[0]
                self.roiw = roi_size[1]
            # update the ROI
            if cross_hair_is_centre:
                self.roiy = int(roi[0]) - self.roih//2
                self.roix = int(roi[1]) - self.roiw//2      
            else:
                self.roiy = int(roi[0])        
                self.roix = int(roi[1])
        else:
            self.roiy = int(roi[0])
            self.roix = int(roi[1])
            self.roih = int(roi[2])
            self.roiw = int(roi[3])
            if cross_hair_is_centre:
                print('Using manual ROI: cross_hair_is_centre = True ignored')

        print(f'{self.channel} ROI set to: top-left corner:(y: {self.roiy}, x: {self.roix}), h: {self.roih} w: {self.roiw}')

        return [self.roiy, self.roix, self.roih, self.roiw]

    def set_polyroi(self, 
                polyroi: np.array=None, 
                threshold: int=None, 
                roi: bool=False) -> None:
        """Set an arbitrary polygon region of interest
        """
        
        if polyroi is not None:            
            self.polyroi = polyroi
            return polyroi
                
        if threshold is None:

            default_backend = mpl.get_backend()
            mpl.use('Qt5Agg')  # need this backend for RoiPoly to work
            
            fig = plt.figure(figsize=(10,10), dpi=80)
            

            if roi:
                _, img, _, _ = self.roi_image()
                extent = [self.roix, self.roix+self.roiw, self.roiy+self.roih, self.roiy] # coordinates of (left, right, bottom, top)
            else:
                img = self.img_ave[self.winy:self.winy+self.winh, self.winx:self.winx+self.winw]
                extent = [self.winx, self.winx+self.winw, self.winy+self.winh, self.winy] # coordinates of (left, right, bottom, top)

            _, roi_img, _, _ = self.roi_image()
            roi_ave = np.nanmean(roi_img)
            roi_std = np.nanstd(roi_img)            
            vmin = roi_ave - (roi_std * 10)
            vmax = roi_ave + (roi_std * 5)          

            plt.imshow(img, origin='upper', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title(f'{self.channel}')

            my_roi = RoiPoly(fig=fig) # draw new ROI in red color
            plt.close()
            # Get the masks for the ROIs
            outline_mask = my_roi.get_mask(self.img_ave) # flip the coordinates
            roi_mask = outline_mask # np.flip(outline_mask, axis=0)

            mpl.use(default_backend)  # reset backend
        else:
            roi_mask = self.img_ave > threshold

        print(f'Number of pixels in {self.channel} mask: {np.sum(roi_mask)}')
        self.polyroi = roi_mask
        return roi_mask

    def get_roi_coords(self, polyroi: bool=True) -> Tuple[List[int], List[int]]:
        """Get the coordinates of the ROI, including the polyroi if set.

        :param polyroi: if True, use the polyroi, defaults to True
        :type polyroi: bool, optional
        :return: Lists of the y and x coordinates
        :rtype: Tuple
        """        
        if polyroi:
            if self.polyroi is None:
                raise ValueError('Error: polyroi not set')
            else:
                y, x = np.where(self.polyroi)
        else:
            ys = list(range(self.roiy,self.roiy+self.roih))
            xs = list(range(self.roix,self.roix+self.roiw))
            y = [y_p for x_p in xs for y_p in ys]
            x = [x_p for x_p in xs for y_p in ys]

        return y, x

    # TODO redefine image stats for use of img_one, img_ave, img_std, img_err
    def image_stats(self, 
                    roi: bool=False, 
                    polyroi: bool=False) -> Tuple[float, float, float, int, Tuple]:
        """Print image statistics
        """
        if roi:
            img_one, img_ave, img_std, img_err = self.roi_image(polyroi)            
        else:
            img_one = self.img_one
            img_ave = self.img_ave
            img_std = self.img_std
            img_err = self.img_err
        
        img_one_mean = np.nanmean(img_one, where=np.isfinite(img_one))
        img_one_std = np.nanstd(img_one, where=np.isfinite(img_one))
        img_std_mean = np.nanmean(img_std, where=np.isfinite(img_ave))
        
        img_ave_mean = np.nanmean(img_ave, where=np.isfinite(img_ave))
        img_ave_std = np.nanstd(img_ave, where=np.isfinite(img_ave))
        img_err_mean = np.nanmean(img_err, where=np.isfinite(img_ave))
        
        # get the pixel coordinates and values
        y,x = self.get_roi_coords(polyroi)

        n_pixels = len(np.isfinite(img_ave.flatten()))

        single_frame_stats = (img_one_mean, img_one_std, img_std_mean)
        averaged_stats = (img_ave_mean, img_ave_std, img_err_mean)

        return single_frame_stats, averaged_stats, n_pixels, (y,x)

class DarkImage(Image):
    """Class for handling Dark Images, inherits Image class.
    """
    def __init__(self, scene_path: Path, product_path: Path, channel: str, img_type: str='drk') -> None:
        Image.__init__(self, scene_path, product_path, channel, img_type)
        self.dark = None
        self.dsnu = None

    def dark_signal(self) -> Tuple:
        """Return the mean dark signal (DARK) and Dark Signal Nonuniformity
        (DSNU)

        :return: DARK, DSNU
        :rtype: Tuple
        """
        self.dark = np.mean(self.img_ave)
        self.dsnu = np.std(self.img_ave)
        return self.dark, self.dsnu
    
class LightImage(Image):
    """Class for handling Light Images, inherits Image class."""
    def __init__(self, scene_path: Path, product_path: Path, channel: str, img_type: str='img', calibration_dir: Path=None) -> None:
        Image.__init__(self, scene_path, product_path, channel, img_type)
        if calibration_dir is not None:            
            self.calibration_dir = calibration_dir
            self.mtx, self.new_mtx, self.dist = self.load_calibration()        
        self.f_length = None
        self.dif_img = None

    def estimate_dark_signal(self) -> Tuple:
        """Estimate the dark signal for the image by averaging the corner
        100 x 100 pixels in each image.

        :return: Dark signal and Dark Standard Deviation
        :rtype: Tuple
        """        
        corner1 = self.img_ave[0:99,0:99]
        corner2 = self.img_ave[-100:-1,0:99]
        corner3 = self.img_ave[0:99,-100:-1]
        corner4 = self.img_ave[-100:-1,-100:-1]
        corners = np.concatenate((corner1, corner2, corner3, corner4))

        dark = np.mean(corners)
        dark_err = np.std(corners) / np.sqrt(len(corners))

        return dark, dark_err

    def dark_subtract(self, 
                      dark_image: Union[float, Tuple[float, float], DarkImage]) -> None:
        
        if isinstance(dark_image, float):            
            dark_signal = dark_image
            dark_noise = 0.0
        elif isinstance(dark_image, Tuple):
            dark_signal = dark_image[0]
            dark_noise = dark_image[1]        
        else:
            dark_signal = dark_image.img_ave
            dark_noise = dark_image.img_err

        # img_one
        self.img_one -= dark_signal
        # self.img_one = np.where(self.img_one < 0.0, np.nan, self.img_one) # set underflow values to NaN
        # img_ave
        self.img_ave -= dark_signal
        # self.img_ave = np.where(self.img_ave < 0.0, np.nan, self.img_ave) # set underflow values to NaN
        # img_std
        self.img_std = np.sqrt((self.img_std)**2 + (dark_noise)**2)    
        # img_err
        self.img_err = np.sqrt((self.img_err)**2 + (dark_noise)**2)
        
        self.units = 'Above-Bias Signal DN'
        print(f'Subtracting dark frame for: {self.camera} ({int(self.cwl)} nm)')

    def flat_field(self, flat_image_dir: Path) -> None:
        """Apply flat field correction to the image

        :param flat_image: Flat Image object
        :type flat_image: Image
        """      
        # look up the flat-field image in the directory.
        flat_ave, flat_err = self.load_flat_field(flat_image_dir)

        lst_one = self.img_one.copy()
        lst_ave = self.img_ave.copy()
        
        # img_one
        self.img_one = self.img_one / flat_ave

        # img_ave
        self.img_ave = self.img_ave / flat_ave

        # img_std

        img_std_r = np.divide(self.img_std, lst_one, out=np.zeros_like(self.img_std), where=lst_one!=0)
        self.img_std = np.abs(self.img_one) * np.sqrt(
                                    (img_std_r)**2 
                                    + (flat_err/flat_ave)**2
                                    )

        # img_err
        img_err_r = np.divide(self.img_err, lst_ave, out=np.zeros_like(self.img_err), where=lst_ave!=0)
        self.img_err = np.abs(self.img_ave) * np.sqrt(
                                    (img_err_r)**2 
                                    + (flat_err/flat_ave)**2
                                    )

        print(f'Flat fielding: {self.camera} ({int(self.cwl)} nm)') 

    def load_flat_field(self, flat_field_dir: Path) -> Tuple[np.array, np.array]:
        """Load flat-field image from the flat field directory

        :param flat_field_dir: Flat-Field directory
        :type flat_field_dir: Path
        :return: _description_
        :rtype: Tuple[Image, Image]
        """        
        
        # get list flatfield of given channel from the flat field directory
        file = Path(flat_field_dir, 'signal', self.channel+'_flatfield_ave.tif')
        if file.exists() is False:
            raise FileNotFoundError(f'Error: no flatfield image found for {self.channel} in {flat_field_dir}')
        
        try:
            img = tiff.TiffFile(file)
        except ValueError:
            print('bad file')
        ff_ave = img.asarray()

        # get list flatfield of given channel from the flat field directory
        file = Path(flat_field_dir, 'noise', self.channel+'_flatfield_err.tif')
        if file.exists() is False:
            raise FileNotFoundError(f'Error: no flatfield error found for {self.channel} in {flat_field_dir}')

        try:
            img = tiff.TiffFile(file)
        except ValueError:
            print('bad file')
        ff_err = img.asarray()

        return ff_ave, ff_err

    def load_calibration(self) -> Tuple:             
        # intrinsic matrix
        mtx_path = Path(
                    self.calibration_dir,
                    'channels', 
                    'intrinsic_matrices',
                    self.channel)
        if mtx_path.exists():
            channel_mtx_path = Path(mtx_path, self.channel+'_imtx').with_suffix('.npy')
            mtx = np.load(channel_mtx_path.resolve())
        else:
            mtx = None

        # new intrinsic matrix
        new_mtx_path = Path(
                    self.calibration_dir,
                    'channels', 
                    'new_intrinsic_matrices',
                    self.channel)
        if new_mtx_path.exists():
            channel_mtx_path = Path(new_mtx_path, self.channel+'_nu_imtx').with_suffix('.npy')
            new_mtx = np.load(channel_mtx_path.resolve())
        else:
            new_mtx = None

        # distortion coefficients
        dist_path = Path(
                    self.calibration_dir,
                    'channels',
                    'distortion_coefficients',
                    self.channel)
        if dist_path.exists():
            channel_dist_path = Path(dist_path, self.channel+'_dist').with_suffix('.npy')
            dist = np.load(channel_dist_path.resolve())
        else:
            dist = None
        
        # TODO flat fielding

        return mtx, new_mtx, dist
    
    def calibration_values(self) -> Dict:
        # output the calibration values of focal length, principal point, and
        # distortion coefficients
        
        # TODO get uncertainty values on these

        # get values from matrix
        pitch = 5.86E-3 # mm
        size = (self.width, self.height)
        params = cv2.calibrationMatrixValues(self.mtx, size, self.width*pitch, self.height*pitch) # note that height and width are swapped compared to opencv specification, but this is the only format that produces the expected focal length and x and y dimensions
        fovx = params[0]
        fovy = params[1]
        focal_length = params[2]
        principal_point = params[3]
        aspect_ratio = params[4]

        # get distortion coefficients
        k1 = self.dist[0][0]
        k2 = self.dist[0][1]
        p1 = self.dist[0][2]
        p2 = self.dist[0][3]
        k3 = self.dist[0][4]

        # calibration dictionary:
        cal_dict = {
            'Field of View X (°)': fovx,
            'Field of View Y (°)': fovy,
            'Focal Length (mm)': focal_length,
            'Principal Point X (mm)': principal_point[0],
            'Principal Point Y (mm)': principal_point[1],
            'Aspect Ratio': aspect_ratio,
            'Radial Dist. k1': k1,
            'Radial Dist. k2': k2,
            'Tangential Dist. p1': p1,
            'Tangential Dist. p2': p2,
            'Radial Dist. k3': k3
        }

        return cal_dict
    
    def align_cali_source(self, source: Image, update_roi: bool=False) -> None:
        """Align the given calibration image to the image, by finding
        the source and sample centre points through Gaussian fitting around
        the ROI coordinates. Return the new cali source.

        :param cali_source: Calibration target image.
        :type cali_source: Image
        """        
        source_img = source.img_ave.copy()
        target_img = self.img_ave.copy()

        # Gaussian blur the images
        target_img = cv2.GaussianBlur(target_img, (5, 5), 0)        
        source_img = cv2.GaussianBlur(source_img, (5, 5), 0)        

        # use window size centred on ROI        
        source_img = source_img[
            source.roiy-source.winh//2:source.roiy+source.winh//2, 
            source.roix-source.winw//2:source.roix+source.winw//2]
        target_img = target_img[
            self.roiy-self.winh//2:self.roiy+self.winh//2, 
            self.roix-self.winw//2:self.roix+self.winw//2]        

        # fit a 2D Gaussian to target_img using the target_img ROI as an initial estimate
        yi, xi = np.mgrid[:self.winh, :self.winw]
        xyi = np.vstack([xi.ravel(), yi.ravel()])
        guess = [np.nanmax(target_img), self.winw//2, self.winh//2, 0.001, 0.001, 0.001]
        pred_params, uncert_cov = opt.curve_fit(self.gauss2d, xyi, target_img.ravel(), p0=guess)

        x0, y0 = pred_params[1], pred_params[2]
        tgt_abs_y0 = y0 + self.roiy-self.winh//2
        tgt_abs_x0 = x0 + self.roix-self.winw//2

        # fit a 2D Gaussian to cali_source_img using the image ROI as an initial estimate
        yi, xi = np.mgrid[:source.winh, :source.winw]
        xyi = np.vstack([xi.ravel(), yi.ravel()])
        guess = [np.nanmax(source_img), source.winw//2, source.winh//2, 0.001, 0.001, 0.001]
        pred_params, uncert_cov = opt.curve_fit(self.gauss2d, xyi, source_img.ravel(), p0=guess)

        x0, y0 = pred_params[1], pred_params[2]
        src_abs_y0 = y0 + source.roiy-source.winh//2
        src_abs_x0 = x0 + source.roix-source.winw//2

        # compute the shift required to align the images
        shift_y = np.round(tgt_abs_y0 - src_abs_y0)
        shift_x = np.round(tgt_abs_x0 - src_abs_x0)

        print(f'   Shifting Vertical by {shift_y} pixels')
        print(f'   Shifting Horizontal by {shift_x} pixels')

        # img_one
        source.img_one = np.roll(source.img_one, int(shift_y), axis=0)
        source.img_one = np.roll(source.img_one, int(shift_x), axis=1)        

        # img_ave
        source.img_ave = np.roll(source.img_ave, int(shift_y), axis=0)
        source.img_ave = np.roll(source.img_ave, int(shift_x), axis=1)

        # img_std
        source.img_std = np.roll(source.img_std, int(shift_y), axis=0)
        source.img_std = np.roll(source.img_std, int(shift_x), axis=1)

        # img_err
        source.img_err = np.roll(source.img_err, int(shift_y), axis=0)
        source.img_err = np.roll(source.img_err, int(shift_x), axis=1)

        # update the cali_source roi and window to match self
        if update_roi:
            self.roiy = int(np.round(tgt_abs_y0)) - self.roih//2
            self.roix = int(np.round(tgt_abs_x0)) - self.roiw//2

        # update the cali_source roi and window to match self
        source.roiy = self.roiy
        source.roix = self.roix
        source.roiw = self.roiw
        source.roih = self.roih
        source.winy = self.winy
        source.winx = self.winx
        source.winh = self.winh
        source.winw = self.winw

        return source
    
    @staticmethod
    def gauss2d(xy, amp, x0, y0, a, b, c):
        x, y = xy
        inner = a * (x - x0)**2 
        inner += 2 * b * (x - x0)**2 * (y - y0)**2
        inner += c * (y - y0)**2
        return amp * np.exp(-inner)

class CalibrationImage(Image):
    """Class for handling Calibration Images, inherits Image class."""
    def __init__(self, source_image: LightImage) -> None:
        self.scene_dir = source_image.scene_dir # input directory
        self.products_dir = source_image.products_dir # output directory
        # TODO check scene directory exists
        self.scene = source_image.scene
        self.channel = source_image.channel
        self.img_type = 'cal'
        self.camera = source_image.camera
        self.emission_angle = source_image.emission_angle
        self.azimuth_angle = source_image.azimuth_angle
        self.phase_angle = source_image.phase_angle
        self.serial = source_image.serial
        self.width = source_image.width
        self.height = source_image.height
        self.cwl = source_image.cwl
        self.fwhm = source_image.fwhm
        self.fnumber = source_image.fnumber
        self.flength = source_image.flength
        self.exposure = source_image.exposure
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.roi = source_image.roi
        self.winx = source_image.winx
        self.winy = source_image.winy
        self.winw = source_image.winw
        self.winh = source_image.winh
        self.polyroi = source_image.polyroi
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_one = source_image.img_one
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.img_err = source_image.img_err
        self.reference_reflectance = None
        self.reference_reflectance_err = None
        self.get_reference_reflectance()

    def get_reference_reflectance(self, 
                                  filename: str='isas_spectralon_reference'):
        # load the reference file
        reference_file = Path('..', '..', 
                              'data', 'calibration', 'reflectance_reference', 
                              filename).with_suffix('.csv')
        data = np.genfromtxt(
                reference_file,
                delimiter=',',
                names=True,
                dtype=float)
        # access the cwl and fwhm
        lo = self.cwl - self.fwhm/2
        hi = self.cwl + self.fwhm/2
        band = np.where((data['wavelength'] > lo) & (data['wavelength'] < hi))
        # set the reference reflectance and error
        self.reference_reflectance = np.mean(data['reflectance'][band])
        self.reference_reflectance_err = np.mean(data['uncertainty'][band]) / np.sqrt(len(band))

    def mask_target(self, clip: float=0.10):
        """Mask the calibration target in the image."""
        # cut dark pixels according to average image
        _, roi_img, _, _ = self.roi_image()
        dark_limit = np.quantile(roi_img, clip)
        mask = self.img_ave > dark_limit

        # apply mask to both average image and single image
        self.img_one = self.img_one * mask
        self.img_ave = self.img_ave * mask
        # # cut bright pixels that exceed a percentile
        # mask = self.img_ave < np.quantile(self.roi_image(), 0.90)
        # self.img_ave = self.img_ave * mask

    def compute_reflectance_coefficients(self):
        """Compute the reflectance coefficients for each pixel of the
        calibration target.
        """
        # img_one
        lst_one = self.img_one.copy()
        out = np.full(self.img_one.shape, np.nan)
        np.divide(self.reference_reflectance, self.img_one, out=out, where=self.img_one!=0)
        self.img_one = out

        # img_ave
        lst_ave = self.img_ave.copy()
        out = np.full(self.img_ave.shape, np.nan)
        np.divide(self.reference_reflectance, self.img_ave, out=out, where=self.img_ave!=0)
        self.img_ave = out

        ref_err_r = self.reference_reflectance_err/self.reference_reflectance
        
        # img_std
        out = np.full(self.img_std.shape, np.nan)
        np.divide(self.img_std, lst_one, out=out, where=lst_one!=0)
        img_std_r = out
        self.img_std = np.abs(self.img_one) * np.sqrt((img_std_r)**2 + (ref_err_r)**2)

        # img_err
        out = np.full(self.img_err.shape, np.nan)
        np.divide(self.img_err, lst_ave, out=out, where=lst_ave!=0)
        img_err_r = out
        self.img_err = np.abs(self.img_ave) * np.sqrt((img_err_r)**2 + (ref_err_r)**2)

        self.units = 'Refl. Coeffs. 1/DN/s'

class ReflectanceImage(Image):
    """Class for handling Reflectance Images, inherits Image class."""
    def __init__(self, source_image: LightImage) -> None:
        self.scene_dir = source_image.scene_dir # input directory
        self.products_dir = source_image.products_dir
        # TODO check scene directory exists
        self.scene = source_image.scene
        self.channel = source_image.channel
        self.img_type = 'rfl'
        self.camera = source_image.camera
        self.emission_angle = source_image.emission_angle
        self.azimuth_angle = source_image.azimuth_angle
        self.phase_angle = source_image.phase_angle
        self.serial = source_image.serial
        self.width = source_image.width
        self.height = source_image.height
        self.cwl = source_image.cwl
        self.fwhm = source_image.fwhm
        self.fnumber = source_image.fnumber
        self.flength = source_image.flength
        self.exposure = source_image.exposure
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.roi = source_image.roi
        self.winx = source_image.winx
        self.winy = source_image.winy
        self.winw = source_image.winw
        self.winh = source_image.winh
        self.polyroi = source_image.polyroi
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_one = source_image.img_one
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.img_err = source_image.img_err

    def calibrate_reflectance(self, cali_source: Union[Image, Tuple], find_shift: bool=False) -> CalibrationImage:


        if isinstance(cali_source, tuple):
            # if tuple, assume it is a calibration coefficient
            cali_coeff = cali_source[1][0] # use the averaged image ROI mean
            # use the averaged image ROI standard deviation - 
            # Note, not the standard error, because when making our reflectance
            # calibration, the undertainty for a given pixel is capture by the
            # spread of the pixel values over the pixels selected, because
            # we will eventually be correcting each individual pixel.
            cali_err = cali_source[1][1]
        else:
            # otherwise, assume it is an Image object
            # apply new image alignment method
            if find_shift:
                cali_source = self.align_cali_source(cali_source)    

            cali_coeff = cali_source.img_ave
            cali_err = cali_source.img_err
        
        lst_one = self.img_one.copy()
        lst_ave = self.img_ave.copy()

        # img_one
        self.img_one = self.img_one * cali_coeff
        # img_ave            
        self.img_ave = self.img_ave * cali_coeff
        self.units = 'Reflectance'

        # img_std
        out = np.full(self.img_std.shape, np.nan)
        np.divide(self.img_std, lst_one, out=out, where=lst_one!=0)
        img_std_r = out
        cali_err_r = cali_err/cali_coeff
        self.img_std = np.abs(self.img_one) * np.sqrt((img_std_r)**2 + (cali_err_r)**2)

        # img_err
        out = np.full(self.img_err.shape, np.nan)
        np.divide(self.img_err, lst_ave, out=out, where=lst_ave!=0)
        img_err_r = out
        cali_err_r = cali_err/cali_coeff
        self.img_err = np.abs(self.img_ave) * np.sqrt((img_err_r)**2 + (cali_err_r)**2)

        return cali_source

    def align_cali_source(self, cali_source: Image) -> None:
        """Align the calibration source to the reflectance image, by finding
        the source and sample centre points through Gaussian fitting around
        the ROI coordinates

        :param cali_source: Calibration target image.
        :type cali_source: Image
        """        
        cali_source_img = cali_source.img_ave.copy()
        refl_target_img = self.img_ave.copy()

        # Gaussian blur the images
        refl_target_img = cv2.GaussianBlur(refl_target_img, (5, 5), 0)
        # also invert the reflectance coefficient image, and remove all NaNs
        cali_source_img = 1.0/ cv2.GaussianBlur(cali_source_img, (5, 5), 0)
        cali_source_img = np.where(np.isfinite(cali_source_img), cali_source_img, 0.0)

        # use window size centred on ROI        
        cali_source_img = cali_source_img[
            cali_source.roiy-cali_source.winh//2:cali_source.roiy+cali_source.winh//2, 
            cali_source.roix-cali_source.winw//2:cali_source.roix+cali_source.winw//2]
        refl_target_img = refl_target_img[
            self.roiy-self.winh//2:self.roiy+self.winh//2, 
            self.roix-self.winw//2:self.roix+self.winw//2]        

        # fit a 2D Gaussian to refl_target_img using the refl_target_img ROI as an initial estimate
        yi, xi = np.mgrid[:self.winh, :self.winw]
        xyi = np.vstack([xi.ravel(), yi.ravel()])
        guess = [np.nanmax(refl_target_img), self.winw//2, self.winh//2, 0.001, 0.001, 0.001]
        pred_params, uncert_cov = opt.curve_fit(self.gauss2d, xyi, refl_target_img.ravel(), p0=guess)

        x0, y0 = pred_params[1], pred_params[2]
        tgt_abs_y0 = y0 + self.roiy-self.winh//2
        tgt_abs_x0 = x0 + self.roix-self.winw//2

        # fit a 2D Gaussian to cali_source_img using the image ROI as an initial estimate
        guess = [np.nanmax(cali_source_img), cali_source.winw//2, cali_source.winh//2, 0.001, 0.001, 0.001]
        pred_params, uncert_cov = opt.curve_fit(self.gauss2d, xyi, cali_source_img.ravel(), p0=guess)

        x0, y0 = pred_params[1], pred_params[2]
        src_abs_y0 = y0 + cali_source.roiy-cali_source.winh//2
        src_abs_x0 = x0 + cali_source.roix-cali_source.winw//2

        # compute the shift required to align the images
        shift_y = np.round(tgt_abs_y0 - src_abs_y0)
        shift_x = np.round(tgt_abs_x0 - src_abs_x0)
        
        cali_source.img_ave = np.roll(cali_source.img_ave, int(shift_y), axis=0)
        cali_source.img_ave = np.roll(cali_source.img_ave, int(shift_x), axis=1)

        cali_source.img_std = np.roll(cali_source.img_std, int(shift_y), axis=0)
        cali_source.img_std = np.roll(cali_source.img_std, int(shift_x), axis=1)

        return cali_source

    def normalise(self, base: Image):
        """Normalise the reflectance image to a base image.
        """
        # uncertainty quantification
        self_err = self.img_std/self.img_ave
        base_err = base.img_std/base.img_ave
        self.img_ave = self.img_ave / base.img_ave
        self.units = 'Normalised Reflectance'
        self.img_std = self.img_ave * np.sqrt(self_err**2 + base_err**2)    

class NormalisedImage(ReflectanceImage):
    """Class for handling Normalised Images, inherits ReflectanceImage class."""
    def __init__(self, source_image: LightImage) -> None:
        ReflectanceImage.__init__(self, source_image)
        self.img_type = 'rfl_nrm'
        self.units = 'Normalised Reflectance'

    def normalise(self, base: Image):
        """Normalise the reflectance image to a base image.
        """
        # uncertainty quantification
        self_err = self.img_std/self.img_ave
        base_err = base.img_std/base.img_ave
        self.img_ave = self.img_ave / base.img_ave
        self.units = 'Normalised Reflectance'
        self.img_std = self.img_ave * np.sqrt(self_err**2 + base_err**2)

class CoAlignedImage(Image):
    def __init__(self,
                    source_image: LightImage,
                    destination_image: LightImage=None,
                    homography: np.ndarray=None,
                    roi: bool=False) -> None:
        self.dir = source_image.dir
        # TODO check scene directory exists
        self.scene = source_image.scene
        self.channel = source_image.channel
        self.img_type = 'geo'
        self.camera = source_image.camera
        self.emission_angle = source_image.emission_angle
        self.azimuth_angle = source_image.azimuth_angle
        self.phase_angle = source_image.phase_angle
        self.serial = source_image.serial
        self.width = source_image.width
        self.height = source_image.height
        self.cwl = source_image.cwl
        self.fwhm = source_image.fwhm
        self.fnumber = source_image.fnumber
        self.flength = source_image.flength
        self.exposure = source_image.exposure
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.roi = roi
        self.polyroi = source_image.polyroi
        self.mask = self.roi_mask(show=False)
        self.points = None
        self.descriptors = None
        self.destination = destination_image
        self.matches = None
        self.match_points = None
        self.homography = homography        

    def roi_mask(self, show: bool=False) -> np.ndarray:
        """Set an opencv mask using the ROI information
        """
        mask = np.zeros((self.width, self.height), dtype="uint8")
        cv2mask = cv2.rectangle(mask,
                            (self.roiy, self.roix),
                            (self.roiy+self.roih, self.roix+self.roiw),
                            255, -1)

        if show:
            # apply the mask to the image
            gray = self.img_ave
            masked = cv2.bitwise_and(gray, gray, mask=cv2mask)
            plt.imshow(masked, cmap='gray')
            plt.show()
        return cv2mask

    def find_points(self, method: str) -> int:
        """Find the points of interest in the image.

        :return: success status
        :rtype: int
        """
        # format for opencv
        if self.roi:
            _, roi_img, _, _ = self.roi_image()
            img = roi_img.astype(np.uint8)
        else:
            img = self.img_ave.astype(np.uint8)

        if method == 'ORB':
            # Initiate the ORB feature detector
            MAX_FEATURES = 1000
            orb = cv2.ORB_create(MAX_FEATURES)
            points, descriptors = orb.detectAndCompute(img, self.mask)
            self.points = points
            self.descriptors = descriptors
        elif method == 'SIFT':
            # Initiate the SIFT feature detector
            sift = cv2.SIFT_create()
            points, descriptors = sift.detectAndCompute(img, self.mask)
            self.points = points
            self.descriptors = descriptors
        return len(self.points)

    def find_matches(self, method: str) -> int:

        # set match method
        if method == 'HAMMING':
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            GOOD_MATCH_PERCENT = 0.80

        # find matches
        d_src = self.descriptors
        d_dest = self.destination.descriptors
        matches = list(matcher.match(d_src, d_dest, None))
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Get feature coordinates in each image
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        p_src = self.points
        p_dest = self.destination.points
        for i, match in enumerate(matches):
            points1[i, :] = p_src[match.queryIdx].pt
            points2[i, :] = p_dest[match.trainIdx].pt
        self.match_points = points1
        self.destination.match_points = points2
        self.matches = matches
        return len(self.matches)

    def show_matches(self) -> None:
        """Show the matches between the two images.
        """
        # format for opencv
        if self.roi:
            _, roi_img, _, _ = self.roi_image()
            src_img = roi_img.astype(np.uint8)
            _, roi_img, _, _ = self.destination.roi_image()
            dest_img = roi_img.astype(np.uint8)
        else:
            src_img = self.img_ave.astype(np.uint8)
            dest_img = self.destination.img_ave.astype(np.uint8)
        # Draw top matches
        imMatches = cv2.drawMatches(src_img, self.points,dest_img, self.destination.points,self.matches, None)
        fig = plt.figure()
        plt.imshow(imMatches)
        plt.show()

    def find_homography(self, method: str) -> None:

        # Find the homography matrix
        p_src = self.match_points
        p_dest = self.destination.match_points
        if method == 'RANSAC':
            homography, _ = cv2.findHomography(p_src, p_dest, cv2.RANSAC)
        self.homography = homography

    def align_images(self) -> None:

        # apply the transform
        query_img = self.img_ave
        height, width = self.img_ave.shape
        hmgr = self.homography
        query_reg = cv2.warpPerspective(self.img_ave, hmgr, (width, height))
        self.img_ave = query_reg
        query_reg = cv2.warpPerspective(self.img_std, hmgr, (width, height))
        self.img_std = query_reg
        self.roix = self.destination.roix
        self.roiy = self.destination.roiy
        self.roiw = self.destination.roiw
        self.roih = self.destination.roih
        # TODO apply to noise image

    def show_alignment(self, overlay: bool=True, error: bool=False, ax: object=None, histo_ax: object=None, roi: bool=False) -> None:

        if roi:
            query_reg = self.roi_image()
            train_img = self.destination.roi_image()
        else:
            query_reg = self.img_ave
            train_img = self.destination.img_ave

        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=2)

        if overlay:
            img = query_reg-train_img
            col_max = max(np.max(query_reg.astype(float)), np.max(train_img.astype(float)))
            src = ax.imshow(img, cmap='RdBu', origin='upper', vmin=-col_max, vmax=col_max)
            im_ratio = query_reg.shape[0] / query_reg.shape[1]
            label = 'Source - Destination'
            cbar = plt.colorbar(src, ax=ax, fraction=0.047*im_ratio, label=label)
        elif error:
            img = np.abs(query_reg-train_img)/train_img
            err = ax.imshow(img, origin='upper')
            im_ratio = query_reg.shape[0] / query_reg.shape[1]
            label = 'Err. % (|S. - D.|/D.)'
            cbar = plt.colorbar(err, ax=ax, fraction=0.047*im_ratio, label=label)

            # add histogram
        counts, bins = np.histogram(img[np.isfinite(img)], bins=128)
        histo_ax.hist(bins[:-1], bins, weights=counts, label=f'{self.camera}: {int(self.cwl)} nm', log=True, fill=False, stacked=True, histtype='step')
        histo_ax.set_xlabel(label)

        if self.img_type == 'rfl':
            cbar.formatter.set_powerlimits((1, 1))
        else:
            cbar.formatter.set_powerlimits((0, 0))

        ax.set_title(f'{self.camera}: {self.cwl} nm')
        if ax is None:
            plt.show()

class GeoCalImage(Image):
    def __init__(self, source_image: LightImage, chkrbrd: Tuple, roi: bool=False) -> None:        
        self.scene_dir = source_image.scene_dir
        self.products_dir = source_image.products_dir        
        self.scene = source_image.scene
        self.channel = source_image.channel
        self.img_type = 'geo'
        self.camera = source_image.camera
        self.emission_angle = source_image.emission_angle
        self.azimuth_angle = source_image.azimuth_angle
        self.phase_angle = source_image.phase_angle
        self.serial = source_image.serial
        self.width = source_image.width
        self.height = source_image.height
        self.cwl = source_image.cwl
        self.fwhm = source_image.fwhm
        self.fnumber = source_image.fnumber
        self.flength = source_image.flength
        self.exposure = source_image.exposure
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.winx = source_image.winx
        self.winy = source_image.winy
        self.winh = source_image.winh
        self.winw = source_image.winw        
        self.roi = roi
        self.polyroi = source_image.polyroi
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_one = self.img2uint8(source_image.img_one)
        self.img_ave = self.img2uint8(source_image.img_ave)        
        self.img_std = self.img2uint8(source_image.img_std)
        self.img_err = self.img2uint8(source_image.img_err)
        self.crows = chkrbrd[0]
        self.ccols = chkrbrd[1]
        self.chkrsize = chkrbrd[2]
        self.all_corners = None
        self.object_points = self.define_calibration_points()
        self.corner_points = self.find_corners()
        self.mtx, self.new_mtx, self.dist, self.rvec, self.tvec = None, None, None, None, None
        self.p_mtx = None
        self.calibration_error = None
        self.f_length = None

    @staticmethod
    def img2uint8(img: np.ndarray) -> np.ndarray:
        """Convert a given image to uint8 format.

        :param img: Input image, assumed in 12-bit depth
        :type img: np.ndarray
        :return: Output image, in uint8 format
        :rtype: np.ndarray
        """        
        # convert to 8-bit        
        img = np.clip(np.floor(img/16), 0, 255).astype(np.uint8)
        return img

    def export_calibration(self):
        """Export the intrinsic and extrinsic channel calibration parameters
        """        
        # intrinsic matrix
        mtx_path = Path(
                    self.scene_dir.parent.parent, 
                    'calibration', 
                    'channels', 
                    'intrinsic_matrices',
                    self.channel)
        mtx_path.mkdir(parents=True, exist_ok=True)                    
        channel_mtx_path = Path(mtx_path, self.channel+'_imtx').with_suffix('.csv')
        np.savetxt(channel_mtx_path.resolve(), self.mtx, delimiter=',')
        np.save(channel_mtx_path.with_suffix('.npy').resolve(), self.mtx)

        # new intrinsic matrix
        new_mtx_path = Path(
                    self.scene_dir.parent.parent, 
                    'calibration', 
                    'channels', 
                    'new_intrinsic_matrices',
                    self.channel)
        new_mtx_path.mkdir(parents=True, exist_ok=True)                    
        channel_mtx_path = Path(new_mtx_path, self.channel+'_nu_imtx').with_suffix('.csv')
        np.savetxt(channel_mtx_path.resolve(), self.new_mtx, delimiter=',')
        np.save(channel_mtx_path.with_suffix('.npy').resolve(), self.new_mtx)

        # distortion coefficients
        dist_path = Path(
                    self.scene_dir.parent.parent,
                    'calibration',
                    'channels',
                    'distortion_coefficients',
                    self.channel)
        dist_path.mkdir(parents=True, exist_ok=True)
        channel_dist_path = Path(dist_path, self.channel+'_dist').with_suffix('.csv')
        np.savetxt(channel_dist_path.resolve(), self.dist, delimiter=',')        
        np.save(channel_dist_path.with_suffix('.npy').resolve(), self.dist)
        # rotation vector
        rvec_path = Path( 
                    self.scene_dir,
                    'calibration',
                    'channels',
                    'rotation_vectors',
                    self.channel)
        rvec_path.mkdir(parents=True, exist_ok=True)
        channel_rvec_path = Path(rvec_path, self.channel+'_rvec').with_suffix('.csv')
        np.savetxt(channel_rvec_path.resolve(), self.rvec, delimiter=',')
        np.save(channel_rvec_path.with_suffix('.npy').resolve(), self.rvec)
        # translation vector        
        tvec_path = Path(
                    self.scene_dir,
                    'calibration',
                    'channels',
                    'translation_vectors',
                    self.channel)
        tvec_path.mkdir(parents=True, exist_ok=True)
        channel_tvec_path = Path(tvec_path, self.channel+'_tvec').with_suffix('.csv')
        np.savetxt(channel_tvec_path.resolve(), self.tvec, delimiter=',')
        np.save(channel_tvec_path.with_suffix('.npy').resolve(), self.tvec)

    def define_calibration_points(self):
        # Define calibration object points and corner locations
        objpoints = np.zeros((self.crows*self.ccols, 3), np.float32)
        objpoints[:,:2] = np.mgrid[0:self.crows, 0:self.ccols].T.reshape(-1, 2)
        objpoints *= self.chkrsize
        return objpoints

    def find_corners(self):
        # Find the chessboard corners
        gray = self.img_ave # np.floor(self.img_ave/16).astype(np.uint8) # note hack of division by 17 to allow for overexposed pixels
        if self.roi:
            gray = gray[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        ret, corners = cv2.findChessboardCorners(gray, (self.crows,self.ccols), flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        self.all_corners = ret

        # refine corner locations
        if self.all_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

            if corners[0][0][0] < corners[-1][0][0]:
                # corners are in the wrong order, flip them
                corners = np.flip(corners, axis=0)

            if self.roi:
                corners[:,:,0]+=self.roiy
                corners[:,:,1]+=self.roix
        else:
            print(f'No corners found for {self.camera} {self.cwl} nm')
            corners = None
        return corners

    def show_corners(self, ax: object=None, corner_roi: bool=False):
        # Draw and display the corners
        gray =self.img_ave # np.floor(self.img_ave/16).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = cv2.drawChessboardCorners(rgb, (self.crows,self.ccols), self.corner_points, self.all_corners)
        if self.roi:
            img = img[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]
        elif corner_roi and self.all_corners:
            # find the roi that bounds the corners
            self.roix = int(np.min(self.corner_points[:,:,0]))
            self.roiy = int(np.min(self.corner_points[:,:,1]))
            self.roiw = int(np.max(self.corner_points[:,:,0])-self.roix)
            self.roih = int(np.max(self.corner_points[:,:,1])-self.roiy)
            pad = int(0.3*self.roiw)
            self.roix = np.clip(self.roix-pad, 0, self.width)
            self.roiy = np.clip(self.roiy-pad, 0, self.height)            
            self.roiw = np.clip(self.roiw, 0, self.width-self.roix-2*pad)+2*pad
            self.roih = np.clip(self.roih, 0, self.height-self.roiy-2*pad)+2*pad
            img = img[self.roiy:self.roiy+self.roih, self.roix:self.roix+self.roiw]                
        ax.imshow(img, origin='upper', cmap='gray')        
        ax.set_title(f'{self.camera}: {self.cwl} nm')
        if ax is None:
            plt.show()

    def project_axes(self, ax: object=None, corner_roi: bool=False):
        # Draw and display an axis on the checkerboard
        # project a 3D axis onto the image
        axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
        axis *= self.chkrsize * 5 # default to 5 square axes
        img = self.img_ave #np.floor(self.img_ave/16).astype(np.uint8)
        
        orig_img = img.copy()

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        orig_img = img.copy()

        error = None

        if self.rvec is not None:
            imgpts, jac = cv2.projectPoints(axis, self.rvec, self.tvec, self.mtx, self.dist)   

            # check the error of the imgpts agains tthe corner_points
            corner_pts_proj, jac = cv2.projectPoints(self.object_points, self.rvec, self.tvec, self.mtx, self.dist)   
            error = cv2.norm(self.corner_points, corner_pts_proj, cv2.NORM_L2)/len(imgpts)

            # draw the axis on the image
            corner = tuple(self.corner_points[0].ravel().astype(np.uint16))
            # undistort                        
            img = cv2.line(img, corner, tuple((imgpts[0].ravel()).astype(np.int64)), (255,0,0), 3)
            img = cv2.line(img, corner, tuple((imgpts[1].ravel()).astype(np.int64)), (0,255,0), 3)
            img = cv2.line(img, corner, tuple((imgpts[2].ravel()).astype(np.int64)), (0,0,255), 3)            

        if self.roi:
            img = img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        elif corner_roi and self.all_corners:
            # find the roi that bounds the corners
            self.roix = int(np.min(self.corner_points[:,:,1]))
            self.roiy = int(np.min(self.corner_points[:,:,0]))
            self.roiw = int(np.max(self.corner_points[:,:,1])-self.roix)
            self.roih = int(np.max(self.corner_points[:,:,0])-self.roiy)
            pad = int(0.1*self.roiw)
            self.roix = np.clip(self.roix-pad, 0, self.width)
            self.roiy = np.clip(self.roiy-pad, 0, self.height)            
            self.roiw = np.clip(self.roiw, 0, self.width-self.roix-2*pad)+2*pad
            self.roih = np.clip(self.roih, 0, self.height-self.roiy-2*pad)+2*pad
            img = img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]

        ax.imshow(img, origin='upper')        
        ax.imshow(orig_img, origin='upper', cmap='Reds', alpha=0.5)        
        ax.set_title(f'{self.camera}: {self.cwl} nm')
        if ax is None:
            plt.show()

        return error

    def camera_intrinsic_properties(self) -> None:
        """Get camera properties from the camera intinsic matrix
        """        
        fovx, fovy, f_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(self.mtx, self.img_ave.shape, self.img_ave.shape[0]*5.86E-3, self.img_ave.shape[1]*5.86E-3)
        # to do put these in the object properties
        self.f_length = f_length
        return fovx, fovy, f_length, principal_point, aspect_ratio
    
    def check_distortion(self, ax: object=None):
        """Apply the calibrated distortion coefficients to the image for
        verification.

        :param ax: _description_, defaults to None
        :type a
        """
        undist_img = cv2.undistort(self.img_ave, self.mtx, self.dist)        
        dif_img = (self.img_ave - undist_img)
        if self.roi:
            undist_img = undist_img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
            dif_img = dif_img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        ax.imshow(dif_img, origin='upper', cmap='RdBu')        
        ax.set_title(f'{self.camera}: {self.cwl} nm')

    def calibrate_camera(self):
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([self.object_points], [self.corner_points], (self.width, self.height), None, None)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
        self.mtx = mtx
        self.new_mtx = new_mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return mtx, new_mtx, dist, rvecs, tvecs

class StereoPair():
    def __init__(self, src_image: LightImage, dst_image: LightImage, calibration_dir: Path) -> None:
        self.src = src_image
        self.dst = dst_image  
        self.calibration_dir = calibration_dir      
        self.pair = f'A{self.src.channel}_B{self.dst.channel}'        
        self.scene_dir = self.src.scene_dir
        self.r_mtx = None
        self.t_mtx = None
        self.f_mtx = None
        self.e_mtx = None
        self.rvec, self.tvec, self.baseline = None, None, None
        self.src_pts = None
        self.src_pt_dsc = None
        self.dst_pts = None
        self.dst_pt_dsc = None
        self.obj_pts = None # ?
        self.matches = None
        self.stereo_err = None
        self.v_alignment = None
        self.src_on_left = None
        self.src_r = None # move to light image property?
        self.dst_r = None # move to light image property?
        self.src_p = None # move to light image property?
        self.dst_p = None # move to light image property?
        self.q_mtx = None
        self.src_elines = None # move to light image property?
        self.dst_elines = None # move to light image property?
        self.stereoMatcher = None
        self.points3D = None

    def load_calibration(self):
        # calibration
        # rotation matrix
        rmtx_path = Path(
            self.calibration_dir, 
            'stereo_pairs', 
            'rotation_matrices',
            self.pair)                         
        pair_rmtx_path = Path(rmtx_path, self.pair+'_rmtx').with_suffix('.npy')
        self.r_mtx = np.load(pair_rmtx_path.resolve())

        # translation vector
        tmtx_path = Path(
            self.calibration_dir,
            'stereo_pairs', 
            'translation_vectors',
            self.pair)                           
        pair_tmtx_path = Path(tmtx_path, self.pair+'_tvec').with_suffix('.npy')
        self.t_mtx = np.load(pair_tmtx_path.resolve())

        # essential matrix
        emtx_path = Path(
            self.calibration_dir,
            'stereo_pairs', 
            'essential_matrices',
            self.pair)                           
        pair_emtx_path = Path(emtx_path, self.pair+'_emtx').with_suffix('.npy')
        self.e_mtx = np.load(pair_emtx_path.resolve())

        # fundamental matrix
        fmtx_path = Path(
            self.calibration_dir,
            'stereo_pairs', 
            'fundamental_matrices',
            self.pair)                           
        pair_fmtx_path = Path(fmtx_path, self.pair+'_fmtx').with_suffix('.npy')
        self.f_mtx = np.load(pair_fmtx_path.resolve())

        self.rvec, self.tvec, self.baseline = self.calibration_values()

    def calibration_values(self) -> Tuple:
        # rvecs
        rvec, _ = cv2.Rodrigues(self.r_mtx)
        rvec = np.rad2deg(rvec).T[0]
        # tvecs
        tvec = self.t_mtx.T[0]
        # baseline
        baseline = np.sqrt(np.sum(tvec**2))

        self.baseline = baseline

        return rvec, tvec, baseline

    def export_calibration(self) -> None:
        """Export the stereo pair calibration parameter, of
        - source -> destination camera rotation matrix
        - source -> destination camera translation matrix
        - source -> destination camera fundamental matrix
        - source -> destination camera essential matrix
        """        
        # calibration
        # rotation matrix
        rmtx_path = Path(
            self.calibration_dir,             
            'stereo_pairs', 
            'rotation_matrices',
            self.pair)
        rmtx_path.mkdir(parents=True, exist_ok=True)                    
        pair_rmtx_path = Path(rmtx_path, self.pair+'_rmtx').with_suffix('.csv')
        np.savetxt(pair_rmtx_path.resolve(), self.r_mtx, delimiter=',')
        np.save(pair_rmtx_path.with_suffix('.npy').resolve(), self.r_mtx)
        # translation vector
        tmtx_path = Path(
            self.calibration_dir, 
            'stereo_pairs', 
            'translation_vectors',
            self.pair)
        tmtx_path.mkdir(parents=True, exist_ok=True)                    
        pair_rmtx_path = Path(tmtx_path, self.pair+'_tvec').with_suffix('.csv')
        np.savetxt(pair_rmtx_path.resolve(), self.t_mtx, delimiter=',')
        np.save(pair_rmtx_path.with_suffix('.npy').resolve(), self.t_mtx)

        # scene
        # fundamental matrix
        fmtx_path = Path(
            self.calibration_dir, 
            'stereo_pairs', 
            'fundamental_matrices',
            self.pair)
        fmtx_path.mkdir(parents=True, exist_ok=True)                    
        pair_fmtx_path = Path(fmtx_path, self.pair+'_fmtx').with_suffix('.csv')
        np.savetxt(pair_fmtx_path.resolve(), self.f_mtx, delimiter=',')
        np.save(pair_fmtx_path.with_suffix('.npy').resolve(), self.f_mtx)
        # essential matrix
        emtx_path = Path(
            self.calibration_dir, 
            'stereo_pairs', 
            'essential_matrices',
            self.pair)
        emtx_path.mkdir(parents=True, exist_ok=True)                    
        pair_emtx_path = Path(emtx_path, self.pair+'_emtx').with_suffix('.csv')
        np.savetxt(pair_emtx_path.resolve(), self.e_mtx, delimiter=',')
        np.save(pair_emtx_path.with_suffix('.npy').resolve(), self.e_mtx)

    def calibrate_stereo(self):
        """Calibrate the given stereo pair of cameras, by using matching points
        in the camera views, and known points in the world space, and
        the camera intrinsic matrices and distortion coefficients.
        Finds the Rotation and Translation vectors, and the essential and
        fundamental matrices.
        Note the Fundamental and Essential matrices are specific to the
        calibration scenes.
        """        
        if self.obj_pts == None:
            self.obj_pts = [self.src.object_points]
        if self.src_pts == None:
            self.src_pts = [self.src.corner_points]
        if self.dst_pts == None:
            self.dst_pts = [self.dst.corner_points]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
            self.obj_pts, 
            self.src_pts, self.dst_pts, 
            self.src.mtx, self.src.dist,
            self.dst.mtx, self.dst.dist, 
            (self.src.width, self.src.height), 
            criteria = criteria, 
            flags = cv2.CALIB_FIX_INTRINSIC)
        
        self.r_mtx = R
        self.t_mtx = T
        self.e_mtx = E
        self.f_mtx = F   
        self.stereo_err = ret

        # check R and T against src and dst t_vecs+r_vecs
        src_r, _ = cv2.Rodrigues(self.src.rvec)
        dst_r, _ = cv2.Rodrigues(self.dst.rvec)
        src_t = self.src.tvec
        dst_t = self.dst.tvec
        r_diff = dst_r - np.matmul(R,src_r)
        t_diff = dst_t - (np.matmul(R,src_t)+T)
        print(f'Rotation Difference: {np.sqrt(np.sum(r_diff**2))}')
        print(f'Translation Difference: {np.sqrt(np.sum(t_diff**2))}')

        return ret    

    def compute_epilines(self, matrix_type: str='F') -> None:
        """Compute epilines for each camera in each corresponding image
        """        
        # self.src_elines = np.matmul(self.e_mtx.T, self.dst_pts.reshape(-1,1,2)).reshape(-1,3)
        # self.dst_elines = np.matmul(self.e_mtx.T, self.src_pts.reshape(-1,1,2)).reshape(-1,3)
        if matrix_type == 'F':
            mtx = self.f_mtx
        elif matrix_type == 'E':
            mtx = self.e_mtx
        else:
            raise ValueError('Matrix type must be "F" or "E"')
        
        # self.src_elines = cv2.computeCorrespondEpilines(self.dst_pts.reshape(-1,1,2),2,mtx.T).reshape(-1,3)
        # self.dst_elines = cv2.computeCorrespondEpilines(self.src_pts.reshape(-1,1,2),1,mtx.T).reshape(-1,3)

        self.src_pts = self.src.corner_points
        self.dst_pts = self.dst.corner_points

        self.src_elines = cv2.computeCorrespondEpilines(self.dst_pts.reshape(-1,1,2),1,mtx.T).reshape(-1,3)
        self.dst_elines = cv2.computeCorrespondEpilines(self.src_pts.reshape(-1,1,2),2,mtx.T).reshape(-1,3)
        return self.src_elines, self.dst_elines

    def draw_epilines(self, view: str='dst', ax: object=None) -> None:
        """Draw epilines in the destination image
        """        
        if view == 'dst':
            r,c = self.dst.img_ave.shape
            img = self.dst.img_ave # (np.floor(self.dst.img_ave)/16).astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            pts = self.dst_pts
            lines = self.dst_elines
            title = f'{self.dst.camera}: {self.dst.cwl} nm'
        elif view == 'src':
            r,c = self.src.img_ave.shape
            if ax.title.get_text() == '':
                img = self.src.img_ave # (np.floor(self.src.img_ave)/16).astype(np.uint8)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            else:
                img = ax.get_images()[0]._A
            pts = self.src_pts
            lines = self.src_elines
            title = f'{self.src.camera}: {self.src.cwl} nm'

        for r,pt in zip(lines,pts):
            color = channel_cols(self.dst.camera, rgb=True) # tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            try:
                img = cv2.line(img, (x0,y0), (x1,y1), color,1)
            except:
                print('stop')
            img = cv2.circle(img,tuple(pt.flatten().astype(int)),5,color,-1)            
        
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(img, origin='upper')
            ax.set_title(title)
            fig.show()
        else:
            ax.imshow(img, origin='upper')
            ax.set_title(title)

    def rectify(self,ax: object=None, roi: bool=False, polyroi: bool=False) -> None:
        """Rectify the source and destination images, using the camera
        intrinsic matrices and distortion coefficients, and the stereo
        pair rotation and translation vectors.
        """    
            
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.src.new_mtx, self.src.dist,
                self.dst.new_mtx, self.dst.dist,
                (self.src.width, self.src.height),
                self.r_mtx, self.t_mtx, alpha=-1, newImageSize=(0,0)) #, flags=cv2.CALIB_ZERO_DISPARITY)
        self.src_r = R1
        self.dst_r = R2
        self.src_p = P1
        self.dst_p = P2
        self.q_mtx = Q

        # set vertical alignment flag
        self.v_alignment = self.dst_p[0,3] == 0.0     

        # set src on left flag - 'source' image must be to left of 'destination'
        # image for disparity finding
        if self.v_alignment:
            if self.tvec[1] < -0.05:
                self.src_on_left = True
            else:
                self.src_on_left = False
        else:
            if self.tvec[0] < -0.05:
                self.src_on_left = True
            else:
                self.src_on_left = False
        
        self.src_map1, self.src_map2 = cv2.initUndistortRectifyMap(
            self.src.new_mtx, self.src.dist, self.src_r, self.src_p, 
            (self.src.width, self.src.height), cv2.CV_32FC1)
        
        self.dst_map1, self.dst_map2 = cv2.initUndistortRectifyMap(
            self.dst.new_mtx, self.dst.dist, self.dst_r, self.dst_p, 
            (self.dst.width, self.dst.height), cv2.CV_32FC1)
        
        src_img = self.src.img_ave.copy() # np.clip(np.floor(self.src.img_ave/16),0,255).astype(np.uint8)        
        dst_img = self.dst.img_ave.copy() # np.clip(np.floor(self.dst.img_ave/16),0,255).astype(np.uint8)        
        
        if polyroi:
            # apply the poly roi mask
            if self.src.polyroi is not None:                
                src_img = src_img * self.src.polyroi
            else:
                print('Warning - no PolyROI defined for source image')
            # if self.dst.polyroi is not None:
            #     dst_img = dst_img * self.dst.polyroi
            # else:
            #     print('Warning - no PolyROI defined for destination image')

        # set all pixels outside the roi to 0
        if roi:
            src_img[:, :self.src.roix] = 0
            src_img[:,self.src.roix+self.src.roiw:] = 0
            src_img[:self.src.roiy,:] = 0
            src_img[self.src.roiy+self.src.roih:,:] = 0

            dst_img[:,:self.dst.roix] = 0
            dst_img[:,self.dst.roix+self.dst.roiw:] = 0
            dst_img[:self.dst.roiy,:] = 0
            dst_img[self.dst.roiy+self.dst.roih:,:] = 0
        
        src_img = cv2.remap(src_img, self.src_map1, self.src_map2, cv2.INTER_LINEAR)
        dst_img = cv2.remap(dst_img, self.dst_map1, self.dst_map2, cv2.INTER_LINEAR)
                 
        self.src_rect = src_img
        self.dst_rect = dst_img

        title = f'{np.array2string(self.tvec, precision=2)} {self.dst.camera}: {self.dst.cwl} nm'
        
        if roi:
            # find new ROIs for rectified images
            # should probably really have a new rectified image class...
            self.src_rect_roi = self.find_rect_roi(self.src_rect)     
            self.dst_rect_roi = self.find_rect_roi(self.dst_rect)
            src_img = src_img[self.src_rect_roi['y']:self.src_rect_roi['y']+self.src_rect_roi['h'], self.src_rect_roi['x']:self.src_rect_roi['x']+self.src_rect_roi['w']]
            dst_img = dst_img[self.dst_rect_roi['y']:self.dst_rect_roi['y']+self.dst_rect_roi['h'], self.dst_rect_roi['x']:self.dst_rect_roi['x']+self.dst_rect_roi['w']]

        if self.dst_elines is not None:
            self.draw_epilines('dst')

        if self.src.camera == self.dst.camera:
            ax.imshow(self.src.img_ave, origin='upper')
            ax.set_title(title)
        else:
            ax.imshow(src_img, origin='upper', cmap='Reds')
            ax.imshow(dst_img, origin='upper', cmap='Blues', alpha=0.5)
            ax.set_title(title)
            if self.v_alignment:
                v_lines = np.arange(0, src_img.shape[1], 32)
                for v_line in v_lines:
                    ax.axvline(x = v_line, color = 'k', linestyle = '-', lw=0.4)             
            else:
                h_lines = np.arange(0, src_img.shape[0], 32)
                for h_line in h_lines:
                    ax.axhline(y = h_line, color = 'k', linestyle = '-', lw=0.4)                    

    def compute_disparity(self, ax: object=None) -> None:
        """Compute a disparity map for the given image pair.
        """ 
        src_img = self.src_rect
        dst_img = self.dst_rect

        # check if uint8
        if src_img.dtype != np.int8:
            if np.nanmax(src_img) > 255:
                src_img = np.clip(np.floor(src_img/16), 0 , 255).astype(np.uint8)
        if dst_img.dtype != np.int8:
            if np.nanmax(dst_img) > 255:
                dst_img = np.clip(np.floor(dst_img/16), 0 , 255).astype(np.uint8)        
        
        if self.v_alignment:
            if self.tvec[1] < -0.05:
                self.src_on_left = True
            else:
                self.src_on_left = False
        else:
            if self.tvec[0] < -0.05:
                self.src_on_left = True
            else:
                self.src_on_left = False

        # check if images are above/below each other
        if self.v_alignment:
            src_img = cv2.rotate(src_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            dst_img = cv2.rotate(dst_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not self.src_on_left:
            src_img = cv2.rotate(src_img, cv2.ROTATE_180)
            dst_img = cv2.rotate(dst_img, cv2.ROTATE_180)
            # src_img = cv2.flip(src_img, 1)
            # dst_img = cv2.flip(dst_img, 1)
        left_img = src_img
        right_img = dst_img
        
        # estimate the disparity
        z_est = 0.8 # 0.8 m
        z_min = z_est - 0.10
        z_max = z_est + 0.05
        f = self.src.new_mtx[0][0]
        b = self.baseline
        d_max = np.round(f*b/(z_min)).astype(np.int16)
        d_min = np.round(f*b/(z_max)).astype(np.int16)
        d_range = (np.round((d_max-d_min)/16)*16).astype(np.int16)

        # StereoSGBM Parameters
        prefiltercap = 5 # 5 - 255
        block_size = 5 # 1 - 255
        min_disp = d_min # -128 - +128
        num_disp = d_range # 16 - 256 (divisble by 16)
        uniqRatio = 15 # 0 - 100
        speck_size = 0 # 0 - 1000
        speck_range = 2 # 0 - 31
        dispDiff = -1 # 0 - 128        

        matcher = cv2.StereoSGBM_create(
            preFilterCap=prefiltercap,
            blockSize=block_size, 
            minDisparity=min_disp,
            numDisparities=num_disp,
            uniquenessRatio=uniqRatio,
            speckleWindowSize=speck_size,
            speckleRange=speck_range,
            disp12MaxDiff=dispDiff,
            P1=8*block_size**2,
            P2=32*block_size**2)
        
        self.stereoMatcher = matcher

        disparity = matcher.compute(left_img, right_img)

        if self.v_alignment:
            disparity = cv2.rotate(disparity, cv2.ROTATE_90_CLOCKWISE)
        if not self.src_on_left:                
            disparity = cv2.rotate(disparity, cv2.ROTATE_180)                
            # disparity = cv2.flip(disparity, 1) 
        
        # self.disparity = disparity
        self.disparity = (disparity/16).astype(np.float32)

        # show the disparityS
        if ax is None:
            disp_roi = self.src_rect_roi
            disp_crop = self.disparity[disp_roi['y']:disp_roi['y']+disp_roi['h'], disp_roi['x']:disp_roi['x']+disp_roi['w']]            
            disp_img = self.src_rect[disp_roi['y']:disp_roi['y']+disp_roi['h'], disp_roi['x']:disp_roi['x']+disp_roi['w']]
            fig, ax = plt.subplots()
        else:
            disp_crop = self.disparity
            disp_img = self.src_rect
        disp = ax.imshow(disp_crop, origin='upper', cmap='gray')
        ax.imshow(disp_img, origin='upper', cmap='Reds', alpha=0.5)

        im_ratio = disp_crop.shape[1]/disp_crop.shape[0]
        cbar = plt.colorbar(disp, ax=ax, fraction=0.047*im_ratio)            
        ax.set_title(f'{self.dst.camera}: {self.dst.cwl} nm')

    def depth_from_disparity(self, ax: object=None, coalign_ax: object=None) -> None:
        """Compute the 3D point cloud from the disparity map, Q matrix and 
        rectified projection matrices.
        """
        # check disparity in correct format    
        if self.disparity.dtype == np.int16:    
            self.disparity = (self.disparity/16).astype(np.float32)
        
        # edit q matrix so that q_mtx[3,4] is always +ve (otherwise will be -ve if T_x is positive)
        print(self.q_mtx)
        if self.q_mtx[3,2] < 0:
            self.q_mtx[3,2] = -self.q_mtx[3,2]

        self.points3D = cv2.reprojectImageTo3D(self.disparity, self.q_mtx, handleMissingValues=True)
        
        # set values at 10,000 to NaN
        self.points3D[self.points3D==10000] = np.nan     
        
        self.depth = self.points3D[:,:,2]

        # flip the parity if needed VERY HACKY NEED A PROPER FIX IN THE Q-MATRIX
        # if np.nanmax(self.depth) < 0:
        #     self.depth = -self.depth
        #     self.points3D = -self.points3D

        # show the depth map
        if ax is None:
            # depth_roi = self.dst_rect_roi
            # depth_crop = self.depth[depth_roi['y']:depth_roi['y']+depth_roi['h'], depth_roi['x']:depth_roi['x']+depth_roi['w']]
            fig, ax = plt.subplots()

        # depth_crop = self.depth
        # disp = ax.imshow(depth_crop, origin='upper', cmap='viridis') #, vmin=0.8*0.8, vmax=0.8*1.2)
        # im_ratio = depth_crop.shape[1]/depth_crop.shape[0]
        # cbar = plt.colorbar(disp, ax=ax, fraction=0.047*im_ratio)
        # _, tvec, _ = self.calibration_values()
        # tvec = np.round(tvec*10)/10
        # ax.set_title(f'{tvec}') # '{self.dst.camera}: {self.dst.cwl} nm')
        # # crop the points
        # # self.points3D = self.points3D[depth_roi['y']:depth_roi['y']+depth_roi['h'], depth_roi['x']:depth_roi['x']+depth_roi['w'],:]

        # show the point cloud as a scatter plot with depth as hue
        # estimate the horizontal image size using the field of view and the maximum depth
        max_depth = 1.0
        hfov = 2*np.arctan(self.dst.width/(2*2133))
        max_width = max_depth*np.tan(hfov/2)
        vfov = 2*np.arctan(self.dst.height/(2*2133))
        max_height = max_depth*np.tan(vfov/2)
    

        disp = ax.scatter(self.points3D[:,:,0], self.points3D[:,:,1], c=self.depth, cmap='viridis', s=0.1)
        # disp = ax.imshow(self.depth, origin='upper', cmap='viridis') #, vmin=0.8*0.8, vmax=0.8*1.2)
        im_ratio = self.depth.shape[1]/self.depth.shape[0]
        # set the axes ratio according to the width and height        
        ax.axes.set_aspect('equal')

        cbar = plt.colorbar(disp, ax=ax, fraction=0.047*im_ratio)
        _, tvec, _ = self.calibration_values()
        tvec = np.round(tvec*10)/10
        ax.set_title(f'{tvec}') # '{self.dst.camera}: {self.dst.cwl} nm')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(+0.4, -0.4)

        if coalign_ax is not None:
            coalign_ax.scatter(self.points3D[:,:,0], self.points3D[:,:,1], c=self.depth, cmap='viridis', s=0.01)
            coalign_ax.set_xlim(-0.5, 0.5)
            coalign_ax.set_ylim(+0.4, -0.4)
            coalign_ax.axes.set_aspect('equal')

        return self.points3D    

    def find_rect_roi(self, rect_img: np.array) -> Dict:
        """Find the Region of Interest of the given rectified image

        :return: _description_
        :rtype: Dict
        """ 
        rect_roi = {}         

        # to do need to balance the offset in the y direction so that the content is at matching epilines

        if rect_img.max() != 0:
            rect_roi['x'] = np.min(np.where(np.sum(rect_img, axis=0)>0))
            rect_roi['y'] = np.min(np.where(np.sum(rect_img, axis=1)>0))
            rect_roi['w'] = np.max(np.where(np.sum(rect_img, axis=0)>0)) - rect_roi['x']
            rect_roi['h'] = np.max(np.where(np.sum(rect_img, axis=1)>0)) - rect_roi['y']
        else:
            rect_roi['x'] = 0
            rect_roi['y'] = 0
            rect_roi['w'] = rect_img.shape[1]
            rect_roi['h'] = rect_img.shape[0]
        return rect_roi 

    def find_features(self, view: str, rectified: bool=False) -> tuple:
        """Find feature points and descriptors in the source or destination image

        :param view: 'source' or 'destination' iamge to perform search over
        :type view: str
        :param rectified: Whether to use the rectified image or not, defaults to False
        :type rectified: bool, optional
        :return: feature points and descriptors
        :rtype: tuple
        """        
        if view == 'source':
            if rectified:
                img = self.src_rect
            else:
                img = self.src.img_ave
        elif view == 'destination':
            if rectified:
                img = self.dst_rect
            else:
                img = self.dst.img_ave
        else:
            raise ValueError('View must be "source" or "destination"')
                
        if img.dtype != np.uint8:
            img = np.floor(img/16).astype(np.uint8)

        # MAX_FEATURES = 1000
        # orb = cv2.ORB_create(MAX_FEATURES)
        # points, descriptors = orb.detectAndCompute(img, None) # TODO allow for an actual mask to be applied, rather than just None

        # initiate SIFT detector
        sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=3,
            contrastThreshold=0.02,
            edgeThreshold=10,
            sigma=1.6            
        )
        # find the keypoints and descriptors with SIFT
        points, descriptors = sift.detectAndCompute(img, None) # TODO allow for an actual mask to be applied, rather than just None

        return points, descriptors

    def find_matches(self, use_corners: bool=False, rectified: bool=False) -> int:
        """Find matching points in the source and destination images

        :param use_corners: Use the corner points to find matches, defaults to False
        :type use_corners: bool, optional
        :return: Number of matching points
        :rtype: int
        """     
        # find points and descriptors
        if use_corners:
            # use the corner points already found in the source and destination images during geometric calibration...
            # but then, what are the descriptors for these points?
            self.src_pts = self.src.corner_points
            self.dst_pts = self.dst.corner_points
            # self.src_pt_dsc = ???
            # self.dst_pt_dsc = ???
        else: # otherwise, perform new feature searches            
            self.src_pts, self.src_pt_dsc = self.find_features('source', rectified)
            self.dst_pts, self.dst_pt_dsc = self.find_features('destination', rectified)
        # find matches
        if (self.src_pt_dsc is not None and self.dst_pt_dsc is not None):
            print(f'# Source Points: {len(self.src_pts)}')
            print(f'# Destination Points: {len(self.dst_pts)}')
            
            # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            GOOD_MATCH_PERCENT = 0.90              
            matches = list(matcher.match(self.src_pt_dsc, self.dst_pt_dsc, None))
            matches.sort(key=lambda x: x.distance, reverse=False)
            numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]    

            # # BFMatcher with default params
            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(self.src_pt_dsc, self.dst_pt_dsc,k=2)
            # # Apply ratio test
            # good = []
            # for m,n in matches:
            #     if m.distance < 0.75*n.distance:
            #         good.append([m])
            # matches = good

        else:
            matches = []

        self.matches = matches

        return len(matches)    

    def draw_matches(self, ax: object=None, rectified: bool=False) -> None:
        """Draw the matching points in the source and destination images
        """       
        if rectified:
            src_img = self.src_rect
            dst_img = self.dst_rect
        else:
            src_img = self.src.img_ave # np.floor(self.src.img_ave/16).astype(np.uint8)            
            dst_img = self.dst.img_ave # np.floor(self.dst.img_ave/16).astype(np.uint8)
        img = cv2.drawMatches(src_img, self.src_pts, dst_img, self.dst_pts, self.matches, None)
        # img = cv2.drawMatchesKnn(src_img, self.src_pts, dst_img, self.dst_pts, self.matches, None)
        if ax is None:
            plt.imshow(img, origin='upper')            
            plt.show()
        else:
            ax.imshow(img, origin='upper')
            ax.set_title(f'{self.dst.camera}: {self.dst.cwl} nm')

    def find_homography(self) -> np.ndarray:
        """Find the homography between the source and destination cameras

        :return: Homography matrix
        :rtype: np.ndarray
        """        
        pass
            
    def find_fundamental_mtx(self, use_corners: bool=False) -> np.ndarray:
        """Find the fundamental matrix between the source and destination
        cameras

        :return: Fundamental matrix
        :rtype: np.ndarray
        """        

        # find points and descriptors
        if use_corners:
            # use the corner points already found in the source and destination images during geometric calibration...
            # but then, what are the descriptors for these points?
            self.src_pts = self.src.corner_points
            self.dst_pts = self.dst.corner_points
            # self.src_pt_dsc = ???
            # self.dst_pt_dsc = ???
        else: # otherwise, perform new feature searches              
            # if self.src_pts is None and self.dst_pts is None:   # this is a mess - sort it out when it's actually needed
            #     self.src_pts, self.src_pt_dsc = self.find_features('source')
            #     self.dst_pts, self.dst_pt_dsc = self.find_features('destination')        
                    # update the matches to only include those that are inliers
            src_pts = np.array([self.src_pts[m.queryIdx].pt for m in self.matches])
            dst_pts = np.array([self.dst_pts[m.trainIdx].pt for m in self.matches])
        try: # using a try statement for logic - very bad practise!
            F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)
            # update the matches to only include those that are inliers
            self.src_pts = self.src_pts[mask.ravel()==1]
            self.dst_pts = self.dst_pts[mask.ravel()==1]
        except:
            print('stop')
            F = None
            mask = None

        self.f_mtx = F
        return F
    
    def find_essential_mtx(self, use_corners: bool=False) -> np.ndarray:
        """Find the essential matrix between the source and destination
        cameras

        :return: Essential matrix
        :rtype: np.ndarray
        """                
        # find points and descriptors
        if use_corners:
            # use the corner points already found in the source and destination images during geometric calibration...
            # but then, what are the descriptors for these points?
            self.src_pts = self.src.corner_points
            self.dst_pts = self.dst.corner_points
            # self.src_pt_dsc = ???
            # self.dst_pt_dsc = ???
        else: # otherwise, perform new feature searches  
            if self.src_pts is None and self.dst_pts is None:          
                self.src_pts, self.src_pt_dsc = self.find_features('source')
                self.dst_pts, self.dst_pt_dsc = self.find_features('destination')    
       
        try: # agaIN, BAD PRACTICE TO USE THIS FOR CONTROL LOGIC
            E, mask = cv2.findEssentialMat( # should be updating these for the rectified images
                self.src_pts,
                self.dst_pts,
                self.src.mtx,
                self.src.dist,
                self.dst.mtx,
                self.dst.dist              
                )
            # update the matches to only include those that are inliers
            self.src_pts = self.src_pts[mask.ravel()==1]
            self.dst_pts = self.dst_pts[mask.ravel()==1]
        except:
            E = None
            mask = None
        self.e_mtx = E
        return E

    def compute3D(self):
        """Compute 3D point locations
        """        

        if len(self.src_pts) == 0:
            # compute p_mtx
            return None
        if len(self.dst_pts) == 0:
            # compute p_mtx
            return None

        try:
            # src_pts = np.array([pt.pt for pt in self.src_pts]).T
            # dst_pts = np.array([pt.pt for pt in self.dst_pts]).T
            src_pts = np.array([self.src_pts[m.queryIdx].pt for m in self.matches]).T
            dst_pts = np.array([self.dst_pts[m.trainIdx].pt for m in self.matches]).T
        except:
            print('stop')

        points4D = cv2.triangulatePoints(
                        self.src_p, 
                        self.dst_p, 
                        src_pts, 
                        dst_pts)
        points3D = cv2.convertPointsFromHomogeneous(points4D.T)        
        self.points3D = points3D.squeeze()
        return points3D

    def plot_3D_points(self, ax: object=None):
        x_pts = self.points3D[:,0]
        y_pts = self.points3D[:,1]
        z_pts = self.points3D[:,2]
        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        color = channel_cols(self.dst.camera)
        ax.scatter3D(x_pts, y_pts, z_pts, color=color)
        title = f'{self.dst.camera}: {self.dst.cwl} nm'
        ax.set_title(title)
        # TODO show the plot if the ax was none

def load_scene(
        scene_path: Path, 
        dark_path: Path=None, 
        flat_path: Path=None,
        product_path: Path=None,
        calibration_path: Path=None,
        img_type: str='img',
        display: bool=True,
        display_dark: bool=False,
        window: Union[bool, str]=True, 
        draw_roi: bool=True,        
        caption: str=None,         
        export_scene: bool=True,
        float32: bool=True, 
        uint16: bool=False, 
        uint8: bool=False, 
        fits: bool=True) -> Dict:
    """Load images of the sample.

    :param scene_path: Directory of sample images
    :type scene_path: Path
    :param dark_path: Directory of dark frames, defaults to None
    :type dark_path: Path, optional
    :param flat_path: Directory of flat field images, defaults to None
    :type flat_path: Path, optional
    :param img_type: Image type, defaults to 'img'
    :type img_type: str, optional
    :param display: Display images, defaults to True
    :type display: bool, optional
    :param roi: Display over the ROI on the image, defaults to False
    :type roi: bool, optional
    :param caption: Caption for the grid of plots, defaults to None
    :type caption: Tuple[str, str], optional
    :param threshold: Threshold for the image display, defaults to (None,None)
    :type threshold: Tuple[float,float], optional
    :return: Dictionary of sample LightImages (units of DN)
    :rtype: Dict
    """
    
    # test scene directory exists
    if not scene_path.exists():
        raise FileNotFoundError(f'Could not find {scene_path}')
    channels = sorted(list(next(os.walk(scene_path))[1]))

    # if products in channels list, then drop products
    if 'products' in channels:
        channels.remove('products')
    
    scene = scene_path.name
    
    scene_imgs = {} # store the scene objects in a dictionary
    dark_imgs = {}

    for channel in channels:
        chnl_scene = LightImage(scene_path, product_path, channel, img_type=img_type, calibration_dir=calibration_path)
        chnl_scene.image_load()
        print(f'Loading {scene}: {chnl_scene.camera} ({int(chnl_scene.cwl)} nm)')

        # dark subtraction
        if isinstance(dark_path, Path):
            dark_smpl = DarkImage(dark_path, product_path, channel)
            dark_smpl.image_load()            
            dark_imgs[channel] = dark_smpl
            # Check exposure times are equal
            light_exp = chnl_scene.exposure
            dark_exp = dark_smpl.exposure
            if not np.isclose(light_exp, dark_exp, atol=0.00001):
                raise ValueError(f'Light and Dark Exposure Times are not equal: {light_exp} != {dark_exp}')
            # subtract the dark frame
            chnl_scene.dark_subtract(dark_smpl)
        elif isinstance(dark_path, dict):
            dark_smpl = dark_path[channel]            
            chnl_scene.dark_subtract(dark_smpl)
        else: # if no dark information is provided, make estimate from the light image corners
            dark_smpl = chnl_scene.estimate_dark_signal()
            chnl_scene.dark_subtract(dark_smpl)

        # flat fielding
        if flat_path is not None:
            chnl_scene.flat_field(flat_path)
            
        scene_imgs[channel] = chnl_scene

    if display:
        display_scene(scene_imgs, scene, statistic='single-frame', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(scene_imgs, scene, statistic='averaged', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(scene_imgs, scene, statistic='single-frame-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(scene_imgs, scene, statistic='averaged-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(scene_imgs, scene, statistic='single-frame-snr', window=window, draw_roi=draw_roi, caption=caption)
        display_scene(scene_imgs, scene, statistic='averaged-snr', window=window, draw_roi=draw_roi, caption=caption)

    if display_dark:
        display_scene(dark_imgs, scene+' Darks', statistic='averaged', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(dark_imgs, scene+' Darks', statistic='averaged-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(dark_imgs, scene+' Darks', statistic='averaged-snr', window=window, draw_roi=draw_roi, caption=caption)

    if export_scene:
        save_scene(scene_imgs, float32=float32, uint16=uint16, uint8=uint8, fits=fits)
        save_scene(dark_imgs, float32=float32, uint16=uint16, uint8=uint8, fits=fits)        

    return scene_imgs

def save_scene(scene_imgs: Dict[str, Image], 
               float32: bool=True, 
               uint16: bool=True, 
               uint8: bool=False, 
               fits: bool=True) -> None:
    """Save the image from each channel in the scene.

    :param scene_imgs: _description_
    :type scene_imgs: _type_
    :param float32: _description_, defaults to True
    :type float32: bool, optional
    :param uint16: _description_, defaults to True
    :type uint16: bool, optional
    :param uint8: _description_, defaults to True
    :type uint8: bool, optional
    :param fits: _description_, defaults to True
    :type fits: bool, optional
    """    

    channels = list(scene_imgs.keys())
    for channel in channels:
        chnl_scene = scene_imgs[channel]
        chnl_scene.save_image(float32=float32, uint16=uint16, uint8=uint8, fits=fits)

def display_scene(
        scene_imgs: Dict[str, Image], 
        scene: str, 
        statistic: str='averaged',
        caption: str=None,
        window: Union[bool, str]=True, 
        draw_roi: bool=True, 
        polyroi: bool=False,
        threshold: float=None,
        vmin: float=None,
        vmax: float=None,
        context: object=None) -> Tuple:
    """Display the signal, noise or SNR images of the scene.

    :param scene_imgs: Dictionary of scene images
    :type scene_imgs: Dict
    """
    channels = list(scene_imgs.keys())
    
    title = f'{scene.capitalize()} {statistic}'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption)

    for channel in channels:
        smpl = scene_imgs[channel]
        if context is not None:
            smpl_cntxt = context[channel]
        else:
            smpl_cntxt = None
        smpl.image_display(
            statistic=statistic, 
            window=window, 
            draw_roi=draw_roi, 
            polyroi=polyroi,
            ax=ax[smpl.camera], 
            histo_ax=ax[8], 
            threshold=threshold,
            vmin=vmin, vmax=vmax,
            context=smpl_cntxt)

    show_grid(fig, ax)
    
    return fig, ax

def show_scene_difference(scene_1: Dict[str, Image], scene_2: Dict[str, Image]):
    channels = list(scene_1.keys())
    title = 'Difference'
    fig, ax = grid_plot(title)
    for channel in channels:
        smpl_1 = scene_1[channel]
        smpl_2 = scene_2[channel]
        _, smpl_1_roi_img, _, _ = smpl_1.roi_image()
        _, smpl_2_roi_img, _, _ = smpl_2.roi_image()
        diff = smpl_1.img_ave/(np.nanmedian(smpl_1_roi_img)) - smpl_2.img_ave/(np.nanmedian(smpl_2_roi_img))        
        # diff = smpl_1.img_ave - smpl_2.img_ave
        smpl_1.dif_img = diff
        smpl_1.image_display(window=True, draw_roi=False, ax=ax[smpl_1.camera], histo_ax=ax[8], statistic='dif_img')
    show_grid(fig, ax)

def align_scenes(
        target_scene: Dict[str, LightImage], 
        source_scene: Dict[str, LightImage], 
        display: bool=True,
        update_rois: bool=False) -> Dict:
    """Align the source scene to the target scene.

    :param target_scene: Dictionary of target scene images
    :type target_scene: Dict
    :param source_scene: Dictionary of source scene images
    :type source_scene: Dict
    :param display: Display the difference between the target and source scenes, defaults to True
    :type display: bool, optional
    :param update_rois: Update the ROI of the target scene with the fitted Gauss coordinates, defaults to False
    :type update_rois: bool, optional
    :return: Dictionary of aligned source scene images
    :rtype: Dict
    """
    channels = list(target_scene.keys())
    aligned_scene = {}

    if display:
        show_scene_difference(target_scene, source_scene)

    for channel in channels:
        target = target_scene[channel]
        source = source_scene[channel]
        print(f'Aligning {target.scene}: {target.camera} ({int(target.cwl)} nm) to {source.scene}: {source.camera} ({int(source.cwl)} nm)')
        align = target.align_cali_source(source, update_roi=update_rois)        
        aligned_scene[channel] = align
    
    if display:
        show_scene_difference(target_scene, aligned_scene)
    
    return aligned_scene

def load_dark_frames(scene: str='sample', roi: bool=False, caption: Tuple[str, str]=None) -> Dict:
    channels = sorted(list(Path('..', 'data', scene).glob('[!.]*')))
    dark_imgs = {} # store the calibration objects in a dictionary
    title = f'{scene} Mean Dark Images'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption[0])
    title = f'{scene} Std. Dev. Dark Images'
    fig1, ax1 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[1])
    for channel_path in channels:
        channel = channel_path.name
        dark_smpl = DarkImage(scene, channel)
        dark_smpl.image_load()
        print(f'Loading {scene}: {dark_smpl.camera} ({int(dark_smpl.cwl)} nm)')
        # show
        dark_smpl.image_display(roi=roi, ax=ax[dark_smpl.camera], histo_ax=ax[8])
        dark_smpl.image_display(noise=True, roi=roi, ax=ax1[dark_smpl.camera], histo_ax=ax1[8])
        dark_imgs[channel] = dark_smpl
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return dark_imgs

def calibrate_channel_reflectance(
        cali_imgs: Dict[str, CalibrationImage], 
        clip: float=0.25, 
        average: bool=False, 
        caption: Tuple[str, str]=None, 
        display: bool=True,
        window: bool=True,
        draw_roi: bool=True) -> Dict:
    """Calibrate the reflectance correction coefficients for images
    of the Spectralon reflectance target.

    :param cali_imgs: Dictionary of LightImages of the Spectralon target
    :type cali_imgs: Dict
    :param caption: Caption for the grid of plots, defaults to None
    :type caption: Tuple[str, str], optional    
    :param clip: clip the image histogram to the given percentile, defaults to None
    :type clip: float, optional
    :return: Dictionary of reflectance correction coefficient CalibrationImages
    :rtype: Dict
    """
    channels = list(cali_imgs.keys())
    cali_coeffs = {}

    for channel in channels:
        # load the calibration target images
        cali = cali_imgs[channel]
        print(f'Finding Reflectance Correction for: {cali.camera} ({int(cali.cwl)} nm)')
        # apply exposure correction
        cali.correct_exposure()
        # compute calibration coefficient image
        cali_coeff = CalibrationImage(cali)
        if clip is not None:
            cali_coeff.mask_target(clip)
        cali_coeff.compute_reflectance_coefficients()
            
        if average:
            cali_coeffs[channel] = cali_coeff.image_stats(roi=True)
        else:
            cali_coeffs[channel] = cali_coeff

    if display:        
        scene = 'Reflectance Calibration Coefficients'
        display_scene(cali_coeffs, scene, statistic='single-frame', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(cali_coeffs, scene, statistic='averaged', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(cali_coeffs, scene, statistic='single-frame-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(cali_coeffs, scene, statistic='averaged-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(cali_coeffs, scene, statistic='single-frame-snr', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(cali_coeffs, scene, statistic='averaged-snr', window=window, draw_roi=draw_roi, caption=caption)            

    return cali_coeffs

def apply_reflectance_calibration(
        scene_imgs: Dict[str, Image], 
        cali_coeffs: Dict[str, CalibrationImage], 
        find_shift: bool=False,
        export_scene: bool=True,
        display: bool=True,
        window: bool=True,
        draw_roi: bool=True,
        caption: Tuple[str,str,str]=None) -> Dict:
    """Apply reflectance calibration coefficients to the sample images.

    :param sample_imgs: Dictionary of LightImage objects (units of DN)
    :type sample_imgs: Dict
    :return: Dictionary of Reflectance Images (units of Reflectance)
    :rtype: Dict
    """
    channels = list(scene_imgs.keys())
    reflectance = {}  
    shift_cali_coeffs = {}  
    if find_shift:
        show_scene_difference(scene_imgs, cali_coeffs)    
    for channel in channels:
        smpl = scene_imgs[channel]
        # apply exposure correction
        smpl.correct_exposure()
        # apply calibration coefficients
        cali_coeff = cali_coeffs[channel]
        refl = ReflectanceImage(smpl)
        shift_cali_coeffs[channel] = refl.calibrate_reflectance(cali_coeff, find_shift=find_shift)
        reflectance[channel] = refl
        scene = refl.scene        

    if find_shift:
        show_scene_difference(scene_imgs, shift_cali_coeffs)

    if export_scene:
        save_scene(reflectance, float32=True, fits=True, uint8=False, uint16=False)

    if display:
        title = f'{scene} Reflectance'
        display_scene(reflectance, title, statistic='single-frame', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(reflectance, title, statistic='averaged', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(reflectance, title, statistic='single-frame-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(reflectance, title, statistic='averaged-noise', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(reflectance, title, statistic='single-frame-snr', window=window, draw_roi=draw_roi, caption=caption)            
        display_scene(reflectance, title, statistic='averaged-snr', window=window, draw_roi=draw_roi, caption=caption)            

    return reflectance

def load_reference_reflectance(refl_imgs, reference_filename: str):
        # load the reference file
        reference_dir = Path('../../data/calibration/reflectance_reference')
        reference_file = Path(reference_dir, reference_filename).with_suffix('.csv')
        data = np.genfromtxt(
                reference_file,
                delimiter=',',
                names=True,
                dtype=float)
        # access the cwl and fwhm
        channels = list(refl_imgs.keys())
        cwls = []
        means = []
        stds = []
        for channel in channels:
            camera = refl_imgs[channel]
            lo = camera.cwl - camera.fwhm/2
            hi = camera.cwl + camera.fwhm/2
            band = np.where((data['wavelength'] > lo) & (data['wavelength'] < hi))
            # set the reference reflectance and error
            cwls.append(camera.cwl)
            means.append(np.mean(data['reflectance'][band]))
            stds.append(np.std(data['reflectance'][band]))
        reference_reflectance = pd.DataFrame({'cwl':cwls, 'reflectance':means, 'error':stds})
        reference_reflectance.sort_values(by='cwl', inplace=True)
        return reference_reflectance

def get_channel_reference_reflectance(cali_coeffs: Dict[str, ReflectanceImage]) -> pd.DataFrame:
    """Get the reference reflectance and error for each channel.

    :param cali_coeffs: Dictionary of CalibrationImage objects for each channel
    :type cali_coeffs: Dict
    :return: Pandas DataFrame of reference reflectance and error for each channel
    :rtype: pd.DataFrame
    """
    ref_val = {}
    ref_err = {}
    cwl = {}
    channels = list(cali_coeffs.keys())
    for channel in channels:
        cali_coeff = cali_coeffs[channel]
        ref_val[channel] = cali_coeff.reference_reflectance
        ref_err[channel] = cali_coeff.reference_reflectance_err
        cwl[channel] = cali_coeff.cwl
    reference = pd.DataFrame({'cwl': cwl, 'reflectance': ref_val, 'error': ref_err}, index=channels)
    reference.sort_values('cwl', inplace=True)
    return reference

def normalise_channel_reflectance(
        target_scene: Dict[str, Image], 
        base_scene: Dict[str, Image],
        display: bool=True) -> Dict[str, Image]:
    """Normalise the target scene to the base scene.
    Target Scene and Base Scene must contain the same camera device numbers.
    Target Scene and Base Scene must have the same units.

    :param target_scene: scene to be normalised
    :type target_scene: Dict[str, Image]
    :param base_scene: scene to use for normalising
    :type base_scene: Dict[str, Image]
    :param display: Display the normalised scene, defaults to True
    :type display: bool, optional
    :return: Normalised Scene
    :rtype: Dict[str, Image]
    """    
    channels = list(target_scene.keys())
    normalised_scene = {}

    # map the target scene keys to the base scene keys
    target_keys = sorted(target_scene.keys())
    base_keys = sorted(base_scene.keys())
    t2b_dict = dict(zip(target_keys, base_keys))

    for channel in channels:
        target = target_scene[channel]
        base = base_scene[t2b_dict[channel]]
        print(f'Normalising {target.scene}: {target.camera} ({int(target.cwl)} nm) to {base.scene}: {base.camera} ({int(base.cwl)} nm)')
        norm = NormalisedImage(target)
        norm.normalise(base)
        normalised_scene[channel] = norm

    if display:
        display_scene(normalised_scene, 'Normalised Scene', statistic='signal')

    return normalised_scene

def set_channel_rois(
        smpl_imgs: Dict[str, Image], 
        roi_size: int=None, 
        roi_dict: Dict=None,
        cross_hair_is_centre: bool=False) -> Dict:
    """Set the region of interest on each channel of the given set of channels.

    :param smpl_imgs: Dictionary of LightImage objects
    :type smpl_imgs: Dict
    :roi_size: Size of the region of interest, defaults to None
    :type roi_size: int, optional
    :roi_dict: Dictionary of region of interest parameters, defaults to None
    :type roi_dict: Dict, optional
    :return: Dictionary of LightImage objects
    :rtype: Dict
    """
    channels = list(smpl_imgs.keys())
    new_roi_dict = {}
    for channel in channels:
        smpl = smpl_imgs[channel]
        if roi_dict is not None:
            roi_params = roi_dict[channel]
        else:
            roi_params = None
        new_roi = smpl.set_roi(
                    roi_size=roi_size, 
                    roi_params=roi_params, 
                    cross_hair_is_centre=cross_hair_is_centre)
        new_roi_dict[channel] = new_roi
    display_rois(smpl_imgs, roi_name='ROI Update', window='roi_centred')
    return new_roi_dict

def set_roi(aligned_imgs: Dict, base_channel: str='6_550', caption: Tuple[str,str]=(None,None)) -> Dict:
    # set region of interest on the base channel
    channels = list(aligned_imgs.keys())
    base = aligned_imgs['6_550']
    base.img_ave = aligned_imgs['6_550'].img_ave.copy()
    base.roi =True
    base.set_polyroi()
    # apply the polyroi to each channel:
    fig, ax = grid_plot('Sample Region of Interest Selection')
    if caption is not None:
        grid_caption(caption[0])
    fig1, ax1 = grid_plot('Sample Region of Interest Selection SNR')
    if caption is not None:
        grid_caption(caption[1])
    vmax = 0.0
    vmin=1.0
    for channel in channels:
        smpl = aligned_imgs[channel]
        smpl.polyroi = base.polyroi
        roi_img = smpl.roi_image(polyroi=True)
        smpl_max = np.max(roi_img[np.isfinite(roi_img)])
        smpl_min = np.min(roi_img[np.isfinite(roi_img)])
        if smpl_max > vmax:
            vmax = smpl_max
        if smpl_min < vmin:
            vmin = smpl_min
    for channel in channels:
        smpl = aligned_imgs[channel]
        smpl.image_display(roi=True,polyroi=True, ax=ax[smpl.camera], vmin=vmin, vmax=vmax, histo_ax=ax[8])
        smpl.image_display(roi=True,polyroi=True, snr=True, ax=ax1[smpl.camera], histo_ax=ax1[8])
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return aligned_imgs

def analyse_roi_reflectance(
        refl_imgs: Dict[str, ReflectanceImage],
        roi_name: str,
        reference_reflectance: pd.DataFrame=None,
        polyroi: bool=True,
        show_spatial_stddev: bool=False,        
        display_roi: bool=False,
        caption: str=None) -> pd.DataFrame:
    """Analyse the reflectance over the Region of Interest, 
    and plot and export the results

    :param refl_imgs: Dictionary of ReflectanceImage objects
    :type refl_imgs: Dict
    :return: Pandas DataFrame of reflectance over the ROI
    :rtype: pd.DataFrame
    """

    # roi stats output
    img_one_means = []
    img_ave_means = []
    img_one_stds = []
    img_ave_stds = []
    img_std_means = []
    img_err_means = []
    img_one_stderrs = []
    img_ave_stderrs = []

    # roi derived stats output
    img_one_snu = []
    img_ave_snu = []
    img_one_snr = []
    img_one_ssnur = []
    img_ave_snr = []
    img_ave_ssnur = []

    # additional information
    channel_coords = {}
    cwls = []
    n_pixs = []
    exposures = []
    cam_nums = []    
    phase_angles = []   
    azimuth_angles = [] 
    emission_angles = []

    channels = list(refl_imgs.keys())
    for channel in channels:
        refl_img = refl_imgs[channel]
        scene = refl_img.scene
        products_dir = refl_img.products_dir
        single_frame_stats, averaged_stats, n_pix, coords = refl_img.image_stats(roi=True, polyroi=polyroi)
        
        img_one_means.append(single_frame_stats[0])
        img_ave_means.append(averaged_stats[0])
        img_one_stds.append(single_frame_stats[1])
        img_ave_stds.append(averaged_stats[1])
        img_std_means.append(single_frame_stats[2])
        img_err_means.append(averaged_stats[2])
        img_one_stderrs.append(single_frame_stats[2] / np.sqrt(n_pix))
        img_ave_stderrs.append(averaged_stats[2] / np.sqrt(n_pix))

        img_one_snu.append(single_frame_stats[1] / single_frame_stats[0])
        img_ave_snu.append(averaged_stats[1] / averaged_stats[0])
        img_one_snr.append(single_frame_stats[0] / single_frame_stats[2])
        img_one_ssnur.append(single_frame_stats[0] / single_frame_stats[1])
        img_ave_snr.append(averaged_stats[0] / averaged_stats[2])
        img_ave_ssnur.append(averaged_stats[0] / averaged_stats[1])

        channel_coords[channel] = coords
        cwl = refl_img.cwl
        cwls.append(cwl)
        n_pixs.append(n_pix)
        exposures.append(refl_img.exposure)
        cam_nums.append(refl_img.camera)
        phase_angles.append(refl_img.phase_angle)
        azimuth_angles.append(refl_img.azimuth_angle)
        emission_angles.append(refl_img.emission_angle)

    results = pd.DataFrame({
        'cwl':cwls, 
        'Phase':phase_angles, 
        'Azimuth': azimuth_angles, 
        'Emission': emission_angles, 
        
        'Mean of Single-Frame ROI': img_one_means,
        'Mean of Averaged ROI': img_ave_means,
        
        'Std. Dev. of Single-Frame ROI': img_one_stds,
        'Std. Dev. of Averaged ROI': img_ave_stds,
        
        'Mean of Single-Frame Noise ROI': img_std_means,
        'Mean of Averaged Noise ROI': img_err_means, 

        'Std. Err. of Mean of Single-Frame ROI': img_one_stderrs, 
        'Std. Err. of Mean of Averaged ROI':img_ave_stderrs, 

                
        'ROI N-pixels': n_pixs, 
        'ROI Single-Frame Spatial Nonuniformity': img_one_snu,
        'ROI Averaged Spatial Nonuniformity': img_ave_snu,

        'Single-Frame SNR <signal>/<noise>': img_one_snr,
        'Single-Frame SNR <signal>/stddev(signal)': img_one_ssnur,

        'Averaged SNR <signal>/<noise>': img_ave_snr,
        'Averaged SNR <signal>/stddev(signal)': img_ave_ssnur,
        
        'Exposure': exposures, 
        'Device': cam_nums}) #, 'reflectance (wt)':wt_means, 'std (wt)':wt_stds})
    
    results.sort_values(by='cwl', inplace=True)

    # prepare output directory    
    rois_dir = Path(products_dir, 'rois')
    rois_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = Path(rois_dir, roi_name)
    roi_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{roi_name}_reflectance_data.csv'
    filepath = Path(roi_dir, filename)
    results.to_csv(filepath)

    # output the raw data for each channel
    for channel in channels:
        refl_img = refl_imgs[channel]
        coords = channel_coords[channel]        
        # construct data Df
        single_frame_pixels = refl_img.img_one[coords[0], coords[1]]
        averaged_pixels = refl_img.img_ave[coords[0], coords[1]]
        stddev_pixels = refl_img.img_std[coords[0], coords[1]]
        stderr_pixels = refl_img.img_err[coords[0], coords[1]]
        pixel_data = pd.DataFrame({
            'x':coords[1], 'y':coords[0], 
            'single-frame-values':single_frame_pixels, 
            'averaged-values': averaged_pixels, 
            'stddev-values': stddev_pixels,
            'stderr-values': stderr_pixels
            })

        roi_data_dir = Path(roi_dir, 'roi_data')
        roi_data_dir.mkdir(parents=True, exist_ok=True)        
        filename = f'{channel}_{roi_name}_raw_data.csv'
        filepath = Path(roi_data_dir, filename)
        pixel_data.to_csv(filepath, index=False)

    if display_roi:
        fig, ax = display_rois(refl_imgs, roi_name=roi_name, polyroi=polyroi)
        # save the figure
        filename = f'{roi_name}_context_gridplot.png'
        filepath = Path(roi_dir, filename)
        fig.savefig(filepath, dpi=300)

        # show the ROI as the full window
        fig, ax = display_scene(refl_imgs, roi_name, statistic='averaged', window='roi', draw_roi=True)        

    fig = plt.figure()
    plt.grid(visible=True)

    if reference_reflectance is not None:
        plt.errorbar(
            x=reference_reflectance.cwl,
            y=reference_reflectance.reflectance,
            yerr=reference_reflectance.error,
            fmt='.--',
            color='g',
            capsize=5.0,
            label='Reference Reflectance ± 1σ'
        )

    if show_spatial_stddev:
        plt.errorbar(
                x=results.cwl,
                y=results['Mean of Single-Frame ROI'],
                yerr=results['Std. Dev. of Single-Frame ROI'],
                fmt='',
                linestyle='',   
                ecolor='b',
                elinewidth=1.0,
                label='±Std. Dev. of Single-Frame ROI',
                capsize=4.0)
        title_sfx = 'Spatial Std. Dev.'
        file_sfx = 'spatial_stddev'
    else:
        plt.errorbar(
                x=results.cwl,
                y=results['Mean of Single-Frame ROI'],
                yerr=results['Std. Err. of Mean of Single-Frame ROI'],
                fmt='',
                linestyle='',
                ecolor='b',
                elinewidth=1.0,
                label='±Std. Err. of Mean of Single-Frame ROI',
                capsize=6.0)
        plt.errorbar(
                x=results.cwl,
                y=results['Mean of Single-Frame ROI'],
                yerr=results['Mean of Single-Frame Noise ROI'],
                fmt='',
                linestyle='',
                ecolor='b',
                elinewidth=1.0,
                label='±Mean of Single-Frame Noise ROI',
                capsize=2.0)
        title_sfx = 'Noise Std. Err.'
        file_sfx = 'noise_stddev'

    if show_spatial_stddev:
        plt.errorbar(
                x=results.cwl,
                y=results['Mean of Averaged ROI'],
                yerr=results['Std. Dev. of Averaged ROI'],
                fmt='',
                linestyle='',   
                ecolor='r',
                elinewidth=1.0,
                label='±Std. Dev. of Averaged ROI',
                capsize=4.0)
        title_sfx = 'Spatial Std. Dev.'
        file_sfx = 'spatial_stddev'

    else:
        plt.errorbar(
                x=results.cwl,
                y=results['Mean of Averaged ROI'],
                yerr=results['Std. Err. of Mean of Averaged ROI'],
                fmt='',
                linestyle='',
                ecolor='r',
                elinewidth=1.0,
                label='±Std. Err. of Mean of Averaged ROI',
                capsize=6.0)
        plt.errorbar(
                    x=results.cwl,
                    y=results['Mean of Averaged ROI'],
                    yerr=results['Mean of Averaged Noise ROI'],
                    fmt='',
                    linestyle='',
                    ecolor='r',
                    elinewidth=1.0,
                    label='±Mean of Averaged Noise ROI',
                    capsize=2.0)
        title_sfx = 'Noise Std. Err.'
        file_sfx = 'noise_stddev'
        
    plt.plot(
        results.cwl,
        results['Mean of Single-Frame ROI'],
        'bo-',
        label=f'Mean of Single-Frame ROI'        
    )

    plt.plot(
        results.cwl,
        results['Mean of Averaged ROI'],
        'ro-',
        label=f'Mean of Averaged ROI'        
    )

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'{scene.capitalize()} Reflectance over ROI {roi_name.capitalize()} ± {title_sfx}')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    filename = f'{roi_name}_reflectance_plot_{file_sfx}.png'
    filepath = Path(roi_dir, filename)
    plt.savefig(filepath, dpi=300)

    if caption is not None:
        grid_caption(caption)        
    

    return results

def set_channel_polyrois(
        smpl_imgs: Dict[str, Image], 
        roi_name: str,
        poly_rois: Dict[str, np.ndarray]=None,
        threshold: int=None, 
        roi: bool=False, 
        display: bool=True) -> Dict:
    """Draw an ROI on each image, and set this as a mask.

    :param smpl_imgs: Dictionary of Image objects for each channel
    :type smpl_imgs: Dict
    :return: Updated Image Objects with new ROI masks
    :rtype: Dict
    """    
    channels = list(smpl_imgs.keys())
    new_poly_rois = {}
    for channel in channels:
        smpl = smpl_imgs[channel]
        if poly_rois is not None:
            polyroi = poly_rois[channel]
        else:
            polyroi = None        
        poly_roi = smpl.set_polyroi(polyroi=polyroi, threshold=threshold, roi=roi)
        new_poly_rois[channel] = poly_roi

    if display:
        display_rois(smpl_imgs, roi_name, polyroi=True)

    return new_poly_rois

def display_rois(smpl_imgs: Dict[str, Image], roi_name: str, window: bool=True, draw_roi: bool=True, polyroi: bool=False) -> None:
    """Display the region of interest of each channel in a grid plot.

    :param smpl_imgs: Dictioanry of images to display
    :type smpl_imgs: Dict
    :param window: Use the default window to display the images, defaults to True
    :type window: bool, optional
    :param draw_roi: Draw the Region of Interest, defaults to True
    :type draw_roi: bool, optional
    :param polyroi: Draw the Polygon Region of Interest, defaults to False
    :type polyroi: bool, optional
    """    
    channels = list(smpl_imgs.keys())
    fig, ax = grid_plot(f'{roi_name.capitalize()} Region of Interest')
    for channel in channels:
        smpl = smpl_imgs[channel]
        smpl.image_display(statistic='averaged', window=window, draw_roi=draw_roi, polyroi=polyroi, ax=ax[smpl.camera], histo_ax=ax[8])
    show_grid(fig, ax)
    return fig, ax

def export_images(smpl_imgs: Dict[str, Image], uint8: bool=False, uint16: bool=False, roi: bool=False) -> None:
    """Export the image stack to tiff

    :param smpl_imgs: _description_
    :type smpl_imgs: Dict
    :param uint8: _description_, defaults to False
    :type uint8: bool, optional
    """    
    channels = list(smpl_imgs.keys())
    for channel in channels:
        smpl = smpl_imgs[channel]
        smpl.roi = roi
        smpl.save_image(uint8=uint8, uint16=uint16, roi=roi)

def grid_plot(title: str=None, projection: str=None):
    cam_ax = {}
    if projection == '3d':
        fig, ax = plt.subplots(3,3, figsize=(FIG_W,FIG_W), subplot_kw=dict(projection='3d'))
    else:
        fig, ax = plt.subplots(3,3, figsize=(FIG_W,FIG_W))
    # TODO update this according to camera number
    cam_ax[2] = ax[0][0] # 400
    cam_ax[5] = ax[0][1] # 950
    cam_ax[6] = ax[0][2] # 550
    cam_ax[4] = ax[1][0] # 735
    cam_ax[8] = ax[1][1] # Histogram
    cam_ax[0] = ax[1][2] # 850
    cam_ax[7] = ax[2][0] # 650
    cam_ax[3] = ax[2][1] # 550
    cam_ax[1] = ax[2][2] # 475
    # cam_ax[8].set_title(f'Non-Zero & Finite Image Histograms')
    fig.suptitle(title)
    return fig, cam_ax

def channel_cols(channel: str, rgb: bool=False) -> Tuple[int,int,int]:
    
    colours = {}
    colours[2] = 'tab:cyan' # 400
    colours[1] = 'tab:blue'# 475
    colours[7] = 'tab:green'# 550
    colours[3] = 'tab:olive'# 550
    colours[6] = 'tab:orange'# 650
    colours[4] = 'tab:red'# 735
    colours[0] = 'tab:purple'# 850
    colours[5] = 'tab:pink'# 950
    # cam_ax[8].set_title(f'Non-Zero & Finite Image Histograms')    
    if rgb:
        colour = np.array(colors.to_rgba(colours[channel])[:-1])*255
    else:
        colour = colours[channel]
    return colour

def show_grid(fig, ax):
    # get individual axis dimensions/ratio
    # ax[8].legend()
    fig.tight_layout()
    ax_h, ax_w = ax[0].bbox.height / fig.dpi, ax[0].bbox.width / fig.dpi
    # update figure size to match
    fig.set_size_inches(FIG_W*ax_h/ax_w, FIG_W)
    # ax[2].set_axis_off()
    # ax[8].set_axis_off()
    fig.tight_layout()
    # fig.show()

def grid_caption(caption_text: str) -> None:
    # add a caption
    cap, cax = plt.subplots(figsize=(FIG_W,0.5))
    cax.set_axis_off()
    cap.text(0.5, 0.5, caption_text, ha='center', va='center')
    cap.tight_layout()

"""Calibration Functions"""

def process_flat_fields(flatfield_scene: Dict[str, Image], display: bool=True) -> Dict[str, Image]:
    """Process flat field above-bias signal mean images to give normalised
    flat fields. Process by first median and then mean filtering the images
    over large (128x128) kernels, then normalising to this filtered image,
    to remove shading effects from flat field.

    :param flatfield_scene: Above-bias signal flat field scene (i.e. dark 
                            frame subtracted)
    :type flatfield_scene: Dict[str, Image]
    :return: Normalised flat fields
    :rtype: Dict[str, Image]
    """    
    channels = list(flatfield_scene.keys())
    for channel in channels:
        ff = flatfield_scene[channel]
        # duplicate the flat field image for median and mean filtering        
        flt_ff = sig.medfilt2d(ff.img_ave, (31,31))
        flt_ff = ndi.gaussian_filter(flt_ff, 64)
        lst_img = ff.img_ave.copy()
        ff.img_ave = ff.img_ave / flt_ff
        ff.img_std = ff.img_ave * np.sqrt(2)*(ff.img_std/np.sqrt(ff.n_imgs)) / lst_img # assume pessimisstically that filtered image error is the same as the raw image error - this is still very small

        # check for local mean of 1.0 across the image at scale of 15x15 pixels
        # in central 50% of image
        ff_check = ndi.uniform_filter(ff.img_ave[ff.height//4:3*ff.height//4, ff.width//4:3*ff.width//4], (15,15))
        print(f'{channel}:')
        print(f'Flat field local mean: {np.mean(ff_check)}')
        print(f'Flat field local std. dev.: {np.std(ff_check)}')

        ff.img_type = 'flatfield'
        ff.units = '1'
        ff.save_image(uint16=False, uint8=False, fits=False)
        scene = ff.scene

    if display:
        display_scene(flatfield_scene, scene, statistic='signal', window=False, draw_roi=False)       

    return flatfield_scene

"""Potentially Redundant Functions"""

def load_dtc_frames(subject: str, channel: str) -> pd.DataFrame:
    # initiliase the variables
    mean = []
    std_t = []
    std_rs = []
    t_exp = []
    n_pix = []
    # find the frames for the given channel
    frame_1s = sorted(list(Path('..', 'data', subject, channel).glob('[!.]*_1_calibration.tif')))
    frame_2s = sorted(list(Path('..', 'data', subject, channel).glob('[!.]*_2_calibration.tif')))
    # check the numbers in each list are equal
    # for each exposure, load image 1, 2 and the dark mean image
    n_steps = len(frame_1s)
    for i in range(n_steps):
        img_1 = Image(subject, channel, frame_1s[i].stem)
        img_1.image_load()
        img_2 = Image(subject, channel, frame_2s[i].stem)
        img_2.image_load()
        if img_1.img_ave.mean() == 1:
            continue
        img_ave = np.mean([img_1.img_ave, img_2.img_ave], axis=0)
        img_ave_mean = np.mean(img_ave)
        img_ave_std = np.std(img_ave)
        img_pair = img_1.img_ave - img_2.img_ave
        img_ave_std_rs = np.std(img_pair) / np.sqrt(2)
        mean.append(img_ave_mean)
        std_t.append(img_ave_std)
        std_rs.append(img_ave_std_rs)
        t_exp.append(float(img_1.exposure))
        n_pix.append(img_1.width * img_1.height)
    # put results in a dataframe
    dtc_data = pd.DataFrame(data={
        'exposure': t_exp,
        'n_pix': n_pix,
        'mean': mean,
        'std_t': std_t,
        'std_rs': std_rs,
    })
    dtc_data = dtc_data.sort_values(by='exposure')
    # fit read noise
    # fit linear to std_rs**2 vs exposure
    fit = np.polyfit(dtc_data['exposure'], dtc_data['std_rs']**2, 1, w=1/dtc_data['std_rs']**2)
    if fit[-1] < 0:
        fit[-1] = 0.0
    read_noise = np.sqrt(fit[-1])

    # fit linear to mean vs exposure for dark current (DN/s)
    fit = np.polyfit(dtc_data['exposure'], dtc_data['mean'], 1)
    dark_current = fit[0]
    bias = fit[1]

    return dtc_data, read_noise, dark_current, bias

def load_ptc_frames(subject: str, channel: str, read_noise: float=None) -> pd.DataFrame:
    # initiliase the variables
    mean = []
    std_t = []
    d_mean = []
    d_dsnu = []
    std_rs = []
    t_exp = []
    n_pix = []
    # find the frames for the given channel
    frame_1s = sorted(list(Path('..', 'data', subject, channel).glob('[!.]*_1_calibration.tif')))
    frame_2s = sorted(list(Path('..', 'data', subject, channel).glob('[!.]*_2_calibration.tif')))
    frame_ds = sorted(list(Path('..', 'data', subject, channel).glob('[!.]*_d_drk.tif')))
    # check the numbers in each list are equal
    # for each exposure, load image 1, 2 and the dark mean image
    n_steps = len(frame_1s)
    for i in range(n_steps):
        img_1 = Image(subject, channel, frame_1s[i].stem)
        img_1.image_load()
        img_2 = Image(subject, channel, frame_2s[i].stem)
        img_2.image_load()
        try:
            drk  = Image(subject, channel, frame_ds[i].stem)
        except:
            print('bad dark')
        drk.image_load()
        # process the images, store the results
        if img_1.img_ave.mean() == 1:
            continue
        img_off = img_1.img_ave - drk.img_ave
        img_off_mean = np.mean(img_off)
        img_off_std = np.std(img_off)
        img_pair = img_1.img_ave - img_2.img_ave
        img_off_std_rs = np.std(img_pair) / np.sqrt(2)
        dark_mean = np.mean(drk.img_ave)
        dark_dsnu = np.std(drk.img_ave)
        mean.append(img_off_mean)
        std_t.append(img_off_std)
        d_mean.append(dark_mean)
        d_dsnu.append(dark_dsnu)
        std_rs.append(img_off_std_rs)
        t_exp.append(float(img_1.exposure))
        n_pix.append(img_1.width * img_1.height)
    # put results in a dataframe
    pct_data = pd.DataFrame(data={
        'exposure': t_exp,
        'n_pix': n_pix,
        'mean': mean,
        'std_t': std_t,
        'std_rs': std_rs,
        'd_mean': d_mean,
        'd_dsnu': d_dsnu
    })
    pct_data = pct_data.sort_values(by='exposure')

    # set as the mean for the highest valued std_t
    full_well = pct_data['mean'][pct_data['std_t'] == pct_data['std_t'].max()].mean()

    if read_noise is None:
        # get read noise DN
        lim = pct_data.index.get_loc(pct_data.index[pct_data['mean'] == pct_data['mean'][pct_data['mean'] < 0.7*full_well].max()][0])

        # fit quadratic to std_t vs mean
        # fit = np.polyfit(pct_data['mean'][0:lim], pct_data['std_t'][0:lim]**2, 2)

        # fit quadratic to std_rs vs mean
        # fit = np.polyfit(pct_data['mean'][0:lim], pct_data['std_rs'][0:lim]**2, 2)
        # if fit[-1] < 0:
        #     fit[-1] = 0.0
        # read_noise = np.sqrt(fit[-1])

        # fit linear to std_rs**2 vs mean
        fit = np.polyfit(pct_data['exposure'][0:lim], pct_data['std_rs'][0:lim]**2, 1)

        if fit[-1] < 0:
            fit[-1] = 0.0
        read_noise = np.sqrt(fit[-1])
        
        # assume read noise of 1.45 DN
        # read_noise = 1.45

    # get read corrected shot noise
    pct_data['std_s'] = np.sqrt(pct_data['std_rs']**2 - read_noise**2)

    # get sensitivity e-/DN
    pct_data['k_adc'] = pct_data['mean'] / pct_data['std_s']**2

    # get mean sensitivity in linear range - 0.05 - 0.95 x Full Well
    lin_range = (pct_data['mean'] < full_well*0.95) & (pct_data['mean'] > full_well*0.05)
    k_adc = pct_data['k_adc'][lin_range].mean()

    # get linearity
    fit = np.polyfit(pct_data['exposure'][lin_range], pct_data['mean'][lin_range], 1, w=1.0/(pct_data['mean'][lin_range])**2)
    offset = fit[1]
    response = fit[0]
    pct_data['linearity'] = 100.0*((pct_data['mean'] - (fit[1]+fit[0]*pct_data['exposure'])) / (fit[1]+fit[0]*pct_data['exposure']))
    lin_min = pct_data['linearity'][lin_range].min()
    lin_max = pct_data['linearity'][lin_range].max()

    # convet dark signal to electrons
    # pct_data['d_mean'] = pct_data['d_mean'] * k_adc
    # pct_data['d_dsnu'] = pct_data['d_dsnu'] * k_adc
    # get SNR
    pct_data['snr'] = pct_data['mean'] / pct_data['std_s']
    pct_data['snr_t'] = pct_data['mean'] / pct_data['std_t']
    # get electron signals
    pct_data['e-'] = pct_data['mean'] * k_adc
    # get electron noise
    pct_data['e-_noise'] = pct_data['std_s'] * k_adc

    return pct_data, full_well, k_adc, read_noise, lin_min, lin_max, offset, response

def get_exposures(smpl_imgs: Dict) -> pd.Series:
    """Show the exposures for each channel of the given set of channels.

    :param smpl_imgs: Dictionary of LightImage objects
    :type smpl_imgs: Dict
    :return: Pandas Series of exposures
    :rtype: pd.Series
    """
    exposures = {}
    channels = list(smpl_imgs.keys())
    for channel in channels:
        smpl = smpl_imgs[channel]
        exposures[channel] = smpl.exposure
    exposures = pd.Series(exposures)
    return exposures

def normalise_reflectance(refl_imgs: Dict, base_channel: str='6_550', caption: str=None) -> Dict:
    """Normalise the reflectance images to the given channel.

    :param refl_imgs: Dictionary of ReflectanceImage objects
    :type refl_imgs: Dict
    :return: Dictionary of ReflectanceImage objects
    :rtype: Dict
    """
    # optionally normalise reflectance images to 550 nm channel
    channels = list(refl_imgs.keys())
    base = refl_imgs['6_550']
    base.img_ave = refl_imgs['6_550'].img_ave.copy()
    fig, ax = grid_plot('Normalised Reflectance')
    if caption is not None:
        grid_caption(caption[0])
    fig1, ax1 = grid_plot('Normalised Reflectance SNR')
    if caption is not None:
        grid_caption(caption[1])
    vmax = 1.0
    vmin = 1.0
    norm_imgs = {}
    for channel in channels:
        refl = refl_imgs[channel]
        norm = ReflectanceImage(refl)
        norm.img_ave = refl.img_ave.copy()
        norm.normalise(base)
        if norm.polyroi is not None:
            roi_img = norm.roi_image(polyroi=True)
        else:
            roi_img = norm.roi_image()
        smpl_max = np.max(roi_img[np.isfinite(roi_img)])
        smpl_min = np.min(roi_img[np.isfinite(roi_img)])
        if smpl_max > vmax:
            vmax = smpl_max
        if smpl_min < vmin:
            vmin = smpl_min
        norm_imgs[channel] = norm
    for channel in channels:
        norm = norm_imgs[channel]
        norm.image_display(roi=True, polyroi=True, ax=ax[norm.camera], histo_ax=ax[8], vmin=vmin, vmax=vmax)
        norm.image_display(roi=True, polyroi=True, snr=True, ax=ax1[norm.camera], histo_ax=ax1[8])
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return norm_imgs

def build_session_directory(session_path: Path) -> Dict:
    """Build the session directory around the given session path.    

    :param session_path: Path of directory containing session notebook
    :type path: Path
    :return session_dict: Dictionary of session paths
    :rtype: Dict
    """    
    # camera system calibration
    cali_path = Path(session_path, 'calibration')
    stereo_cali_path = Path(cali_path, 'stereo_pairs')
    channel_cali_path = Path(cali_path, 'channels')
    cali_path.mkdir(parents=True, exist_ok=True)
    stereo_cali_path.mkdir(parents=True, exist_ok=True)
    channel_cali_path.mkdir(parents=True, exist_ok=True)

    # scenes - to be called for each 'scene' in directory
    scenes_path = Path(session_path, 'scenes')
    scenes_path.mkdir(parents=True, exist_ok=True)

    session_dict = { # really this should be some properties of a new session object
        'session_path': session_path,
        'cali_path': cali_path,
        'stereo_cali_path': stereo_cali_path,
        'channel_cali_path': channel_cali_path,
        'scenes_path': scenes_path
    }

    return session_dict

def build_calibration_directory(
        session_dict: Dict, 
        calibration_src: str) -> Dict:
    """Build the calibration directory in the session directory, with symlinks
    built pointing to the given calibration source directory. This should be 
    of the format [instrument name]_[date of acquistion].

    :param session_dict: Dictionary of paths for the session directory
    :type session_dict: Dict
    :param calibration_src: Name of the source calibration directory
    :type calibration_src: str
    :return session_dict: Updated dictionary of paths for the session directory
    :rtype: Dict
    """        
    # single channels
    # build an distortion coefficients directory - if there
    src_cali_path = Path('..', '..', 'data', 'calibration', calibration_src)
    dist_path = Path(session_dict['channel_cali_path'], 'distortion_coefficients')
    src_dist_path = Path(src_cali_path, 'channels', 'distortion_coefficients')
    if src_dist_path.exists():
        if not dist_path.exists():
            try:
                dist_path.symlink_to(src_dist_path.resolve(), target_is_directory=True)    
            except OSError:
                copytree(src_dist_path.resolve(), dist_path)
    # build an intrinsic matrix directory - if there    
    i_mtx_path = Path(session_dict['channel_cali_path'], 'intrinsic_matrices')
    src_i_mtx_path = Path(src_cali_path, 'channels', 'intrinsic_matrices')
    if src_i_mtx_path.exists():
        if not i_mtx_path.exists():
            try:
                i_mtx_path.symlink_to(src_i_mtx_path.resolve(), target_is_directory=True) 
            except OSError:
                copytree(src_i_mtx_path.resolve(), i_mtx_path)
    # build a new intrinsic matrix directory - if there    
    new_i_mtx_path = Path(session_dict['channel_cali_path'], 'new_intrinsic_matrices')
    src_nu_i_mtx_path = Path(src_cali_path, 'channels', 'new_intrinsic_matrices')
    if src_nu_i_mtx_path.exists():
        if not new_i_mtx_path.exists():
            try:
                new_i_mtx_path.symlink_to(src_nu_i_mtx_path.resolve(), target_is_directory=True)    
            except OSError:
                copytree(src_nu_i_mtx_path.resolve(), new_i_mtx_path)
    # build a flat field directory and symlink to the calibration folder - if there
    flat_field_path = Path(session_dict['channel_cali_path'], 'flat_fields')
    src_flat_field_path = Path(src_cali_path, 'channels', 'flat_fields')
    if src_flat_field_path.exists():
        if not flat_field_path.exists():
            try:
                flat_field_path.symlink_to(src_flat_field_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_flat_field_path.resolve(), flat_field_path)
        # check that the path has been assigned the directory

    # stereo pairs
    # build a rotation matrices directory and symlink to the calibration folder - if there
    rmtx_path = Path(session_dict['stereo_cali_path'], 'rotation_matrices')
    src_rmtx_path = Path(src_cali_path, 'stereo_pairs', 'rotation_matrices')
    if src_rmtx_path.exists():
        if not rmtx_path.exists():
            try:
                rmtx_path.symlink_to(src_rmtx_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_rmtx_path.resolve(), rmtx_path)
        # check that the path has been assigned the directory
    # build a translation vector directory and symlink to the calibration folder - if there
    tvec_path = Path(session_dict['stereo_cali_path'], 'translation_vectors')
    src_tvec_path = Path(src_cali_path, 'stereo_pairs', 'translation_vectors')
    if src_tvec_path.exists():
        if not tvec_path.exists():
            try:
                tvec_path.symlink_to(src_tvec_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_tvec_path.resolve(), tvec_path)
        # check that the path has been assigned the directory
    # build a fundamental matrix directory and symlink to the calibration folder - if there
    fmtx_path = Path(session_dict['stereo_cali_path'], 'fundamental_matrices')
    src_fmtx_path = Path(src_cali_path, 'stereo_pairs', 'fundamental_matrices')
    if src_fmtx_path.exists():
        if not fmtx_path.exists():
            try:
                fmtx_path.symlink_to(src_fmtx_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_fmtx_path.resolve(), fmtx_path)
        # check that the path has been assigned the directory
    # build a essential matrix directory and symlink to the calibration folder - if there
    emtx_path = Path(session_dict['stereo_cali_path'], 'essential_matrices')
    src_emtx_path = Path(src_cali_path, 'stereo_pairs', 'essential_matrices')
    if src_emtx_path.exists():
        if not emtx_path.exists():
            try:
                emtx_path.symlink_to(src_emtx_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_emtx_path.resolve(), emtx_path)
        # check that the path has been assigned the directory

    # update the path dictionary and return
    session_dict['i_mtx_path'] = i_mtx_path
    session_dict['new_mtx_path'] = new_i_mtx_path
    session_dict['flat_field_path'] = flat_field_path
    session_dict['rmtx_path'] = rmtx_path
    session_dict['tvec_path'] = tvec_path

    return session_dict

def build_scene_directory(
        scene_name: str,
        src_scene_path: str, 
        src_dark_path: str,         
        scenes_path: str) -> None:
    """Build the directory tree in the scenes directory for the given scene and
      dark frame data in the data directory.

    :param scene_name: name of the scene directory
    :type scene_name: str
    :param src_scene_path: path of the scene direcotry in the 'data' directory
    :type src_scene_path: Path
    :param src_dark_path: path of the dark frame direcotry in the 'data' directory
    :type src_dark_path: Path
    :param scenes_dir: path of the scenes direcotry in the current session
    :type scenes_dir: Path
    """        
    # build a scene directory under this name
    scene_path = Path(scenes_path, scene_name)
    scene_path.mkdir(exist_ok=True)
    # build the calibration directory
    cali_path = Path(scene_path, 'calibration')
    cali_path.mkdir(exist_ok=True)
    # build the channels and stereo_pairs calibration directories
    cali_channels_path = Path(cali_path, 'channels')
    cali_channels_path.mkdir(exist_ok=True)
    cali_stereo_pairs_path = Path(cali_path, 'stereo_pairs')
    cali_stereo_pairs_path.mkdir(exist_ok=True)    
    # build a dark frame directory and symlink to the dark path
    dark_path = Path(cali_channels_path, 'dark_frames')
    if src_dark_path.exists():
        if not dark_path.exists():
            try:
                dark_path.symlink_to(src_dark_path.resolve(), target_is_directory=True)    
            except OSError:
                copytree(src_dark_path.resolve(), dark_path)

    else:
        raise ValueError(f'Dark frame directory does not exist: {src_dark_path}')
    # build the raw data directory
    raw_path = Path(scene_path, 'raw')
    if src_scene_path.exists():
        if not raw_path.exists():
            try:
                raw_path.symlink_to(src_scene_path.resolve(), target_is_directory=True)
            except OSError:
                copytree(src_scene_path.resolve(), raw_path)
    else:
        raise ValueError(f'Scene directory does not exist: {src_scene_path}')
    # build the products directory
    prod_path = Path(scene_path, 'products')
    prod_path.mkdir(exist_ok=True)

    # # make a symlink to the camera_config.csv file
    # camera_config_path = Path(src_scene_path, '..', 'camera_config.csv').resolve()
    # if camera_config_path.exists():
    #     camera_config_link = Path(scene_path, '..', '..', 'calibration', 'camera_config.csv').resolve()
    #     if not camera_config_link.exists():
    #         try:
    #             camera_config_link.symlink_to(camera_config_path, target_is_directory=False)
    #         except OSError:
    #             copy(camera_config_path, camera_config_link)
    # else:
    #     raise ValueError(f'Camera config file does not exist: {camera_config_path}')

    return raw_path, dark_path

def checkerboard_calibration(geocs: Dict, chkrbrd: Tuple, roi: bool=False, caption: str=None) -> Dict:
    """Find the checkerboard corners in the geometric calibration image for each
    channel.

    :param geocs: Dictionary of calirbation images
    :type geocs: Dict
    :param caption: Caption for the gridplot, defaults to None
    :type caption: str, optional
    :return: _description_
    :rtype: Dict
    """    
    channels = list(geocs.keys())
    corners = {}
    fig, ax = grid_plot('Checkerboard point finding')
    if caption is not None:
        grid_caption(caption)
    for channel in channels:
        cali_src = geocs[channel]
        print(f'Finding Checkerboard Corners for: {cali_src.camera} ({cali_src.cwl} nm)')
        src = GeoCalImage(cali_src, chkrbrd=chkrbrd, roi=roi)
        # target_points = src.define_calibration_points()
        # found_points = src.find_corners()
        src.show_corners(ax=ax[src.camera], corner_roi=roi)
        corners[channel] = src
    show_grid(fig, ax)
    return corners

def calibrate_homography(geocs: Dict, caption: Tuple[str, str]=None) -> Dict:
    """Calibrate the homography matrix for images of the geometric calibration
    target.

    :param geocs: Dictionary of calirbation images
    :type geocs: Dict
    :param caption: Caption for the gridplot, defaults to None
    :type caption: Tuple[str, str], optional
    :return: Dictionary of geometric correction LightImages
    :rtype: Dict
    """
    channels = list(geocs.keys())
    destination = '6_550'
    cali_dest = geocs[destination]
    coals = {}
    fig, ax = grid_plot('Target Alignment: Overlay')
    if caption is not None:
        grid_caption(caption[0])
    fig1, ax1 = grid_plot('Target Alignment: Error')
    if caption is not None:
        grid_caption(caption[1])
    for channel in channels:
        cali_src = geocs[channel]
        src = CoAlignedImage(cali_src, roi=False)
        print(f'Calibrating Homography for: {src.camera} ({src.cwl} nm)')
        n_ps = src.find_points('ORB')
        print(f'...{n_ps} source points found')
        dest = CoAlignedImage(cali_dest, roi=False)
        n_pd = dest.find_points('ORB')
        print(f'...{n_pd} destination points found')
        src.destination = dest
        n_m = src.find_matches('HAMMING')
        print(f'...{n_m} matches found')
        # src.show_matches()
        src.find_homography('RANSAC')
        src.align_images()
        src.show_alignment(overlay=True, error=False, ax=ax[src.camera], histo_ax=ax[8], roi=True)
        src.show_alignment(overlay=False, error=True, ax=ax1[src.camera], histo_ax=ax1[8], roi=True)
        coals[channel] = src
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return coals

def apply_coalignment(smpl_imgs: Dict, coals: Dict, caption: Tuple[str, str]=None) -> Dict:
    """Apply homography coalignment to the sample images.

    :param smpl_imgs: Dictionary of sample images (units of Reflectance)
    :type smpl_imgs: Dict
    :param coals: Dictionary of geometric correction LightImages
    :type coals: Dict
    :return: Dictionary of co-aligned sample images (units of Reflectance)
    :rtype: Dict
    """
    channels = list(smpl_imgs.keys())
    destination = '6_550' # TODO should read this from the coals object.
    sample_dest = smpl_imgs[destination]
    fig, ax = grid_plot('Sample Alignment: Overlay')
    if caption:
        grid_caption(caption[0])
    fig1, ax1 = grid_plot('Sample Alignment: Error')
    if caption:
        grid_caption(caption[1])
    aligned_refl = {}
    for channel in channels:
        sample_src = smpl_imgs[channel]
        sample_coal = coals[channel]
        src = CoAlignedImage(
                    sample_src,
                    destination_image=sample_dest,
                    homography=sample_coal.homography,
                    roi=False)
        src.align_images()
        src.show_alignment(overlay=True, error=False, ax=ax[src.camera], histo_ax=ax[8], roi=True)
        src.show_alignment(overlay=False, error=True, ax=ax1[src.camera], histo_ax=ax1[8], roi=True)
        aligned_refl[channel] = ReflectanceImage(src) # TODO make this a generic image type
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return aligned_refl

# def load_geometric_calibration(
#         scene_path: Path, 
#         dark_path: Path, 
#         display: bool=True, 
#         caption: str=None) -> Dict:
#     """Load the geometric calibration images.

#     :param scene_path: Directory of geometric calibration images
#     :type scene_path: Path
#     :param dark_path: Directory of dark frames
#     :type dark_path: Path
#     :param display: Display images, defaults to True
#     :type display: bool, optional
#     :param caption: Caption for the grid of plots, defaults to None
#     :type caption: Tuple[str, str], optional
#     :return: Dictionary of geometric correction LightImages
#     :rtype: Dict
#     """    
#     # test scene directory exists
#     if not scene_path.exists():
#         raise FileNotFoundError(f'Could not find {scene_path}')
#     # scene = scene_path.name
#     # channels = sorted(list(scene_path.glob('[!.]*')))
#     channels = sorted(list(next(os.walk(scene_path))[1]))


#     # channels = sorted(list(scene_path.glob('[!._]*')))
#     geocs = {}
#     if display:
#         fig, ax = grid_plot('Geometric Calibration Target')
#         if caption is not None:
#             grid_caption(caption)
#     for channel in channels:
#         # load the geometric calibration images
#         geoc = LightImage(scene_path, channel)
#         geoc.image_load()
#         print(f'Loading Geometric Target for: {geoc.camera} ({geoc.cwl} nm)')
#         # load the geometric calibration dark frames
#         dark_geoc = DarkImage(dark_path, channel)
#         dark_geoc.image_load()
#         # subtract the dark frame
#         geoc.dark_subtract(dark_geoc)
#         # show
#         if display:
#             geoc.image_display(roi=False, ax=ax[geoc.camera], histo_ax=ax[8])
#         geocs[channel] = geoc
#     if display:
#         show_grid(fig, ax)
#     return geocs

# def load_reflectance_calibration(
#         scene_path: Path, 
#         dark_path: Path, 
#         product_path: Path,
#         display: bool=True, 
#         roi: bool=False, 
#         caption: str=None, 
#         threshold: Tuple[float,float]=(None,None),
#         save_image: bool=False) -> Dict:
    
#     # test scene directory exists
#     if not scene_path.exists():
#         raise FileNotFoundError(f'Could not find {scene_path}')
#     scene = scene_path.name
#     channels = sorted(list(next(os.walk(scene_path))[1]))
#     # if products in channels list, then drop products
#     if 'products' in channels:
#         channels.remove('products')

#     # set up plots
#     title = f'{scene} Images'
#     if display:
#         fig, ax = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[0])
#         title = f'{scene} Images Noise'
#         fig1, ax1 = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[1])
#         title = f'{scene} Image SNR'
#         fig2, ax2 = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[2])

#     cali_imgs = {} # store the calibration objects in a dictionary
#     for channel in channels:

#         # load the calibration target images
#         cali = LightImage(scene_path, product_path, channel)
#         cali.image_load()
        
#         # load the calibration target dark frames
#         dark_cali = DarkImage(dark_path, product_path, channel)
#         dark_cali.image_load()
        
#         # Check exposure times are equal
#         light_exp = cali.exposure
#         dark_exp = dark_cali.exposure
#         if light_exp != dark_exp:
#             raise ValueError(f'Light and Dark Exposure Times are not equal: {light_exp} != {dark_exp}')
        
#         # subtract the dark frame
#         cali.dark_subtract(dark_cali)
        
#         # show
#         if display:
#             cali.image_display(window=True, draw_roi=roi, ax=ax[cali.camera], histo_ax=ax[8], threshold=threshold[0])
#             cali.image_display(window=True, draw_roi=roi, noise=True, ax=ax1[cali.camera], histo_ax=ax1[8], threshold=threshold[1])
#             cali.image_display(window=True, draw_roi=roi, snr=True, ax=ax2[cali.camera], histo_ax=ax2[8], threshold=threshold[1])
        
#         cali_imgs[channel] = cali
#         if save_image:
#             cali.save_image(float32=True, uint8=True, uint16=True, fits=True)
    
#     # show plots
#     show_grid(fig, ax)
#     show_grid(fig1, ax1)
#     show_grid(fig2, ax2)
#     if caption is not None:
#         grid_caption(caption)

#     return cali_imgs


    # def save_image(self, uint8: bool=False, uint16: bool=False):
    #     """Save the average and error images to TIF files"""
    #     # TODO update to control conversion to string, and ensure high precision for exposure time

    #     # prepare metadata
    #     metadata={
    #         'scene': self.scene,
    #         'image-type': self.img_type,
    #         'camera': self.camera,
    #         'serial': self.serial,
    #         'cwl': self.cwl,
    #         'fwhm': self.fwhm,
    #         'f-number': self.fnumber,
    #         'f-length': self.flength,
    #         'exposure': self.exposure,
    #         'units': self.units,
    #         'n_imgs': self.n_imgs
    #     }
    #     cwl_str = str(int(self.cwl))
    #     cam_num = str(self.camera)        

    #     # prepare filename and output directory
    #     name = 'mean'
    #     filename = cam_num+'_'+cwl_str+'_'+name+'_'+self.img_type
    #     product_dir = self.products_dir
    #     product_dir.mkdir(parents=True, exist_ok=True)

    #     tiffs_dir = Path(product_dir, 'tiffs')
    #     fits_dir = Path(product_dir, 'fits')
    #     tiffs_dir.mkdir(parents=True, exist_ok=True)
    #     fits_dir.mkdir(parents=True, exist_ok=True)
        
    #     if self.roi:
    #         out_img = self.roi_image(self.polyroi)
    #         err_img = self.roi_std(self.polyroi)
    #     else:
    #         out_img = self.img_ave
    #         err_img = self.img_std

    #     # out_img = np.clip(out_img, 0.0, None) # clip to ensure non-negative

    #     if uint8:
    #         # convert to 8-bit
    #         out_img = np.floor(out_img/16).astype(np.uint8)
    #     elif uint16:
    #         # convert to 16-bit
    #         out_img = np.floor(out_img).astype(np.uint16)
    #     else:
    #         out_img = out_img.astype(np.float32) # set data type to float 32

    #     # write camera properties to TIF using ImageJ metadata        
    #     img_file =str(Path(tiffs_dir, filename).with_suffix('.tif'))
    #     tiff.imwrite(img_file, out_img, imagej=True, metadata=metadata)
    #     print(f'Mean image written to {img_file}')

    #     # if FITS is true, save to FITs file also
    #     img_file =str(Path(fits_dir, filename).with_suffix('.fits'))

    #     hdu = fitsio.PrimaryHDU(out_img)
    #     hdu.header['scene'] = self.scene
    #     hdu.header['type'] = self.img_type
    #     hdu.header['camera'] = int(self.camera)
    #     hdu.header['serial'] = int(self.serial)
    #     hdu.header['cwl'] = self.cwl
    #     hdu.header['fwhm'] = self.fwhm
    #     hdu.header['f-number'] = self.fnumber
    #     hdu.header['f-length'] = self.flength
    #     hdu.header['exposure'] = self.exposure
    #     hdu.header['units'] = self.units
    #     hdu.writeto(img_file, overwrite=True)

    #     # error image
    #     name = 'error'
    #     filename = cam_num+'_'+cwl_str+'_'+name+'_'+self.img_type
    #     img_file =str(Path(tiffs_dir, filename).with_suffix('.tif'))
    #     # write camera properties to TIF using ImageJ metadata
    #     # clip to ensure nonzero
    #     # err_img = np.clip(err_img, 0.0, None)
    #     # set data type to float 32
    #     err_img = err_img.astype(np.float32) 
    #     tiff.imwrite(img_file, err_img, imagej=True, metadata=metadata)
    #     print(f'Error image written to {img_file}')

    #     # if FITS is true, save to FITs file also
    #     img_file =str(Path(fits_dir, filename).with_suffix('.fits'))
    #     hdu = fitsio.PrimaryHDU(err_img)
    #     hdu.header['scene'] = self.scene
    #     hdu.header['type'] = self.img_type
    #     hdu.header['camera'] = int(self.camera)
    #     hdu.header['serial'] = int(self.serial)
    #     hdu.header['cwl'] = self.cwl
    #     hdu.header['fwhm'] = self.fwhm
    #     hdu.header['f-number'] = self.fnumber
    #     hdu.header['f-length'] = self.flength
    #     hdu.header['exposure'] = self.exposure
    #     hdu.header['units'] = self.units
    #     hdu.writeto(img_file, overwrite=True)

# def analyse_flat_fields(
#         flat_fields: Dict, 
#         scene_path: Path, 
#         dark_path: Path, 
#         display: bool=True, 
#         roi: bool=False, 
#         caption: str=None, 
#         threshold: Tuple[float,float]=(None,None),
#         save_image: bool=False) -> Dict:
#     """Load images of the flat-field.

#     :param flat_fields: Directory of flat-field images
#     :type flat_fields: Dict
#     :param scene_path: Directory of flat-field images
#     :type scene_path: Path
#     :param dark_path: Directory of dark frames
#     :type dark_path: Path
#     :param display: Display images, defaults to True
#     :type display: bool, optional
#     :param roi: Display over the ROI on the image, defaults to False
#     :type roi: bool, optional
#     :param caption: Caption for the grid of plots, defaults to None
#     :type caption: Tuple[str, str], optional
#     :param threshold: Threshold for the image display, defaults to (None,None)
#     :type threshold: Tuple[float,float], optional
#     :return: Dictionary of sample LightImages (units of DN)
#     :rtype: Dict
#     """
#     # test scene directory exists
#     if not scene_path.exists():
#         raise FileNotFoundError(f'Could not find {scene_path}')
#     scene = scene_path.name
#     # channels = sorted(list(scene_path.glob('[!.]*')))
#     channels = sorted(list(next(os.walk(scene_path))[1]))
#     # if products in channels list, then drop products
#     if 'products' in channels:
#         channels.remove('products')
#     ff_imgs = {} # store the flat field mean images in a dictionary
#     title = f'{scene} Median ROIs'
#     if display:
#         fig, ax = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[0])
#         fig1, ax1 = grid_plot('Flat Fields')
#         if caption is not None:
#             grid_caption(caption[0])

#     ff_roll_uniformity = {}
#     ff_uniformity = {}

#     # load spectralon example
#     test_path = Path(scene_path.parent, 'spectralon')
#     test_dark_path = Path(dark_path.parent, 'spectralon_dark')
#     spectralon = load_reflectance_calibration(test_path, test_dark_path, display=True, roi=True, save_image=False)

#     for channel in channels:
#         ff = flat_fields[channel]
#         # get change in mean with each additional position
#         n_positions = ff.n_imgs
#         ff_stk = []

#         smpl = spectralon[channel] # reference target

#         ff_roll_uniformity[channel] = []
#         for i in np.arange(0, n_positions):
#             ff_roll = LightImage(scene_path, channel, img_type='ave')
#             ff_roll.image_load(n_imgs = i+1, mode='median')
#             dark = DarkImage(dark_path, channel)
#             dark.image_load()
#             ff_roll.dark_subtract(dark)

#             # apply the flat field to the reference target
#             ff_roll.img_ave = smpl.img_ave / ff_roll.img_ave

#             # ff_stk.append(ff_roll.roi_image())
#             ff_roll_uniformity[channel].append(100.0* (1 - np.nanstd(ff_roll.roi_image()) / np.nanmean(ff_roll.roi_image())))

#         # Check exposure times are equal
#         light_exp = ff.exposure
#         dark_exp = dark.exposure
#         if light_exp != dark_exp:
#             raise ValueError(f'Light and Dark Exposure Times are not equal: {light_exp} != {dark_exp}')
     
#         # apply the flat field to the reference target
#         smpl.img_ave = smpl.img_ave / ff.img_ave
#         # copy flat field ROI to the smpl ROI
#         smpl.roi = ff.roi
#         smpl.roix = ff.roix
#         smpl.roiy = ff.roiy
#         smpl.roiw = ff.roiw
#         smpl.roih = ff.roih

#         # log the uniformity of the flat-fielded image
#         ff_uniformity[channel] = 100.0* (1 - np.nanstd(smpl.roi_image()) / np.nanmean(smpl.roi_image()))

#         if display:
#             smpl.image_display(roi=roi, ax=ax[smpl.camera], histo_ax=ax[8], threshold=threshold[0])
#             ff.image_display(roi=roi, ax=ax1[ff.camera], histo_ax=ax1[8], threshold=threshold[1])
#         # if save_image:
#         #     ff.save_image(uint16=False)

#     if display:
#         show_grid(fig, ax)
#         show_grid(fig1, ax1)

#     return ff_roll_uniformity, ff_uniformity

# def process_flat_fields(
#         scene_path: Path, 
#         dark_path: Path, 
#         display: bool=True, 
#         roi: bool=False, 
#         caption: str=None, 
#         threshold: Tuple[float,float]=(None,None),
#         save_image: bool=False) -> Dict:
#     """Load images of the flat-field.

#     :param scene_path: Directory of flat-field images
#     :type scene_path: Path
#     :param dark_path: Directory of dark frames
#     :type dark_path: Path
#     :param display: Display images, defaults to True
#     :type display: bool, optional
#     :param roi: Display over the ROI on the image, defaults to False
#     :type roi: bool, optional
#     :param caption: Caption for the grid of plots, defaults to None
#     :type caption: Tuple[str, str], optional
#     :return: Dictionary of sample LightImages (units of DN)
#     :rtype: Dict
#     """
#     # test scene directory exists
#     if not scene_path.exists():
#         raise FileNotFoundError(f'Could not find {scene_path}')
#     scene = scene_path.name
#     # channels = sorted(list(scene_path.glob('[!.]*')))
#     channels = sorted(list(next(os.walk(scene_path))[1]))
#     # if products in channels list, then drop products
#     if 'products' in channels:
#         channels.remove('products')
#     ff_imgs = {} # store the flat field mean images in a dictionary
#     title = f'{scene} Median ROIs'
#     if display:
#         fig, ax = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[0])
#         title = f'{scene} Noise ROIs'
#         fig1, ax1 = grid_plot(title)
#         if caption is not None:
#             grid_caption(caption[1])

#     for channel in channels:
#         ff = LightImage(scene_path, channel, img_type='ave')
#         ff.image_load(mode='median')
#         print(f'Loading {scene}: {ff.camera} ({int(ff.cwl)} nm)')
#         dark = DarkImage(dark_path, channel)
#         dark.image_load()
#         # subtract the dark frame
#         ff.dark_subtract(dark)

#         # Check exposure times are equal
#         light_exp = ff.exposure
#         dark_exp = dark.exposure
#         if light_exp != dark_exp:
#             raise ValueError(f'Light and Dark Exposure Times are not equal: {light_exp} != {dark_exp}')

#         # normalise the flat field to the maximum value in the ROI, after Gaussian blurring with a 3x3 kernel
#         ff_roi = ff.roi_image()
#         ff_roi = cv2.GaussianBlur(ff_roi, (3,3), 0)
#         ff.img_ave = ff.img_ave / ff_roi.max()
#         ff.img_std = ff.img_std / ff_roi.max()
#         ff.img_type = 'flat-field'
#         ff.units = '1'

#         ff_imgs[channel] = ff
        
#         # show
#         if display:
#             ff.image_display(roi=roi, ax=ax[ff.camera], histo_ax=ax[8], threshold=threshold[0])
#             ff.image_display(roi=roi, noise=True, ax=ax1[ff.camera], histo_ax=ax1[8], threshold=threshold[1])

#         if save_image:
#             ff.save_image(uint16=False)
#     if display:
#         show_grid(fig, ax)
#         show_grid(fig1, ax1)

#     return ff_imgs