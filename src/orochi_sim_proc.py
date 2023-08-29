"""A library of functions for processing OROCHI Simulator Images.

Roger Stabbins
Rikkyo University
21/04/2023
"""
from pathlib import Path
import os
from collections import OrderedDict
import cv2
import dict2xml
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import numpy as np
import pandas as pd
from roipoly import RoiPoly, MultiRoi
import scipy
import tifffile as tiff
from typing import Tuple, Dict
import orochi_sim_ctrl as osc

FIG_W = 10 # figure width in inches

class Image:
    """Super class for handling image import and export.
    """
    def __init__(self, scene_path: Path, channel: str, img_type: str, roi: bool=False) -> None:
        """Initialise properties. Most of these are populated during image load.
        """
        self.dir = Path(scene_path, channel)
        # TODO check subject directory exists
        self.subject = scene_path.name
        self.channel = channel
        self.img_type = img_type
        self.camera = None
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
        self.roi = roi
        self.polyroi = None
        self.units = ''
        self.n_imgs = None
        self.img_ave = None
        self.img_std = None    

    def image_load(self, n_imgs: int=None, mode: str='mean') -> None:
        """Load images from the subject directory for the given type,
        populate properties, and compute averages and standard deviation.
        """
        # get list of images of given type in the subject directory
        files = list(self.dir.glob('*'+self.img_type+'.tif'))
        # set n_imgs
        if n_imgs == None:
            self.n_imgs = len(files)
        else:
            self.n_imgs = n_imgs
        self.units = 'Raw DN'
        img_list = []
        if files == []:
            raise FileNotFoundError(f'Error: no {self.img_type} images found in {self.dir}')
        for f, file in enumerate(files[:self.n_imgs]):
            try:
                img = tiff.TiffFile(file)
            except:
                print('bad file')
            img_arr = img.asarray()
            img_list.append(img_arr)
            meta = img.imagej_metadata
            self.camera = self.check_property(self.camera, meta['camera'])
            self.serial = self.check_property(self.serial, meta['serial'])
            self.width = self.check_property(self.width, img_arr.shape[0])
            self.height = self.check_property(self.height, img_arr.shape[1])
            self.cwl = self.check_property(self.cwl, meta['cwl'])
            self.fwhm = self.check_property(self.fwhm, meta['fwhm'])
            self.fnumber = self.check_property(self.fnumber, meta['f-number'])
            self.flength = self.check_property(self.flength, meta['f-length'])
            self.exposure = self.check_property(self.exposure, meta['exposure'])
            try:
                self.roix = self.check_property(self.roix, meta['roix'])
                self.roiy = self.check_property(self.roiy, meta['roiy'])
                self.roiw = self.check_property(self.roiw, meta['roiw'])
                self.roih = self.check_property(self.roih, meta['roih'])
            except KeyError:
                # read the camera_config file to get the ROI
                camera_info = osc.load_camera_config()
                cam_name = f'DMK 33GX249 {int(self.serial)}'
                cam_props = camera_info[cam_name]
                self.roix = cam_props['roix']
                self.roiy = cam_props['roiy']
                self.roiw = cam_props['roiw']
                self.roih = cam_props['roih']

        img_stk = np.dstack(img_list)
        # average stack
        if mode == 'mean':
            self.img_ave = np.mean(img_stk, axis=2)
        elif mode == 'median':
            self.img_ave = np.median(img_stk, axis=2)
        # std. stack
        self.img_std = np.std(img_stk, axis=2)
        print(f'Loaded {f+1} images ({self.img_type}) for: {self.camera} ({int(self.cwl)} nm)')

    def roi_image(self, polyroi: bool=False) -> np.ndarray:
        """Returns the region of interest of the image.
        """
        roi_img = self.img_ave[self.roix:self.roix+self.roiw,self.roiy:self.roiy+self.roih]

        if polyroi:
            roi_img = np.where(self.polyroi, roi_img, np.nan)
        return roi_img

    def roi_std(self, polyroi: bool=False) -> np.ndarray:
        """Returns the region of interest of the error image.
        """
        roi_img = self.img_std[self.roix:self.roix+self.roiw,
                            self.roiy:self.roiy+self.roih]

        if polyroi:
            roi_img = np.where(self.polyroi, roi_img, np.nan)
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

        self.img_ave = self.img_ave / self.exposure
        self.img_std = self.img_std / self.exposure # assume exposure err. negl.
        self.units = 'DN/s'

    def set_polyroi(self) -> None:
        """Set an arbitrary polygon region of interest
        """

        if self.roi:
            img = self.roi_image()
        else:
            img = self.img_ave
        default_backend = mpl.get_backend()
        mpl.use('Qt5Agg')  # need this backend for RoiPoly to work
        fig = plt.figure(figsize=(10,10), dpi=80)
        plt.imshow(img, origin='upper')
        plt.title('Draw ROI')

        my_roi = RoiPoly(fig=fig) # draw new ROI in red color
        plt.close()
        # Get the masks for the ROIs
        outline_mask = my_roi.get_mask(img)
        roi_mask = outline_mask # np.flip(outline_mask, axis=0)

        mpl.use(default_backend)  # reset backend
        self.polyroi = roi_mask

    def image_stats(self, polyroi: bool=False) -> None:
        """Print image statistics
        """
        if polyroi:
            img = self.roi_image()[self.polyroi]
        if polyroi:
            std = self.roi_std()[self.polyroi]
        img_ave_mean = np.nanmean(img, where=np.isfinite(img))
        img_ave_std = np.nanstd(img, where=np.isfinite(img))
        img_std_mean = np.nanmean(std, where=np.isfinite(img))

        # standard error on the mean


        # weighted average
        ma_img = np.ma.array(img, mask=(np.isnan(img) * ~np.equal(std, 0.0))) # mask nans
        # mask where standard deviation is zero
        ma_std = np.ma.array(std, mask=(np.isnan(img) * ~np.equal(std, 0.0)))
        img_ave_wt_mean, wt_sum = np.ma.average(ma_img, weights=1.0/ma_std**2, returned=True)
        img_ave_wt_std = np.sqrt(1.0/wt_sum)

        img_ave_wt_var = np.ma.average((ma_img - img_ave_wt_mean)**2, weights=1.0/ma_std**2)
        img_ave_wt_std = np.sqrt(img_ave_wt_var)

        return img_ave_mean, img_ave_std, img_std_mean, img_ave_wt_mean, img_ave_wt_std

    def image_display(self,
                      ax: object=None, histo_ax: object=None,
                      noise: bool=False, snr: bool=False,
                      threshold: float=None,
                      roi: bool=False, polyroi: bool=False,
                      vmin: float=None, vmax: float=None) -> None:
        """Display the image mean and standard deviation in one frame.
        """
        # set the size of the window
        if ax is None:
            fig, axs = plt.subplots(2,1,figsize=(FIG_W/2, FIG_W))
            ax = axs[0]
            histo_ax = axs[1]
            fig.suptitle(f'Subject: {self.subject} ({self.img_type})')

        if roi:
            img_ave = self.roi_image(polyroi)
            img_std = self.roi_std(polyroi)
        else:
            img_ave = self.img_ave
            img_std = self.img_std

        if noise:
            img = img_std
            label = self.units+ ' Noise'
        elif snr:
            out = np.full(img_ave.shape, np.nan)
            np.divide(img_ave, img_std, out=out, where=img_ave!=0)
            img = out
            label = self.units+' SNR'
        else:
            img = img_ave
            label = self.units

        if threshold:
            img = np.where(img < threshold, np.nan, img)

        ave = ax.imshow(img, origin='upper', vmin=vmin, vmax=vmax)
        im_ratio = img.shape[0] / img.shape[1]
        cbar = plt.colorbar(ave, ax=ax, fraction=0.047*im_ratio, label=label)

        if ('Noise' in label):
            cbar.formatter.set_powerlimits((0, 0))
        elif ('DN' in label):
            cbar.formatter.set_scientific(False)
        elif ('SNR' in label):
            cbar.formatter.set_scientific(False)
        elif ('Reflectance' in label):
            cbar.formatter.set_scientific(False)
        else:
            cbar.formatter.set_powerlimits((0, 0))
        ax.set_title(f'Device {self.camera} ({int(self.cwl)} nm)')

        # add histogram
        counts, bins = np.histogram(img[np.nonzero(np.isfinite(img))], bins=128)
        histo_ax.hist(bins[:-1], bins, weights=counts,
                      label=f'{int(self.cwl)} nm ({self.camera})',
                      log=True, fill=False, stacked=True, histtype='step')
        histo_ax.set_xlabel(label)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax

        # TODO add histograms
        # TODO add method for standard deviation image

    def save_tiff(self, save_stack: bool=False):
        """Save the average and error images to TIF files"""
        # TODO update to control conversion to string, and ensure high precision for exposure time
        metadata={
            'subject': self.subject,
            'image-type': self.img_type,
            'camera': self.camera,
            'serial': self.serial,
            'cwl': self.cwl,
            'fwhm': self.fwhm,
            'f-number': self.fnumber,
            'f-length': self.flength,
            'exposure': self.exposure,
            'units': self.units,
            'n_imgs': self.n_imgs
        }
        cwl_str = str(int(self.cwl))
        print(self.dir)
        # average image
        name = 'mean'
        filename = cwl_str+'_'+name+'_'+self.img_type
        img_file =str(Path(self.dir, filename).with_suffix('.tif'))
        # write camera properties to TIF using ImageJ metadata
        out_img = self.img_ave # .astype(np.float32)
        tiff.imwrite(img_file, out_img, imagej=True, metadata=metadata)
        print(f'Mean image written to {img_file}')

        # error image
        name = 'error'
        filename = cwl_str+'_'+name+'_'+self.img_type
        img_file =str(Path(self.dir, filename).with_suffix('.tif'))
        # write camera properties to TIF using ImageJ metadata
        out_img = self.img_std.astype(np.float32)
        tiff.imwrite(img_file, out_img, imagej=True, metadata=metadata)
        print(f'Error image written to {img_file}')


class DarkImage(Image):
    """Class for handling Dark Images, inherits Image class.
    """
    def __init__(self, subject: str, channel: str, img_type: str='drk') -> None:
        Image.__init__(self,subject, channel, img_type)
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
    def __init__(self, subject: str, channel: str, img_type: str='img') -> None:
        Image.__init__(self,subject, channel, img_type)

    def dark_subtract(self, dark_image: DarkImage) -> None:
        lst_ave = self.img_ave.copy()
        self.img_ave -= dark_image.img_ave
        # quadrture sum the image noise with the dark signal noise
        lght_err = self.img_std/lst_ave
        drk_err = dark_image.img_std/dark_image.img_ave
        self.img_std = self.img_ave * np.sqrt((lght_err)**2 + (drk_err)**2)
        self.units = 'Above-Bias Signal DN'
        print(f'Subtracting dark frame for: {self.camera} ({int(self.cwl)} nm)')


class CalibrationImage(Image):
    """Class for handling Calibration Images, inherits Image class."""
    def __init__(self, source_image: LightImage) -> None:
        self.dir = source_image.dir
        # TODO check subject directory exists
        self.subject = source_image.subject
        self.channel = source_image.channel
        self.img_type = 'cal'
        self.camera = source_image.camera
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
        self.polyroi = source_image.polyroi
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.reference_reflectance = None
        self.reference_reflectance_err = None
        self.get_reference_reflectance()

    def get_reference_reflectance(self):
        # load the reference file
        reference_file = Path('spectralon_reference.csv')
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
        self.reference_reflectance_err = np.std(data['reflectance'][band])

    def mask_target(self, clip: float=0.10):
        """Mask the calibration target in the image."""
        # cut dark pixels
        dark_limit = np.quantile(self.roi_image(), clip)
        mask = self.img_ave > dark_limit
        self.img_ave = self.img_ave * mask
        # # cut bright pixels that exceed a percentile
        # mask = self.img_ave < np.quantile(self.roi_image(), 0.90)
        # self.img_ave = self.img_ave * mask

    def compute_reflectance_coefficients(self):
        """Compute the reflectance coefficients for each pixel of the
        calibration target.
        """
        lst_ave = self.img_ave.copy()
        out = np.full(self.img_ave.shape, np.nan)
        np.divide(self.reference_reflectance, self.img_ave, out=out, where=self.img_ave!=0)
        self.img_ave = out
        out = np.full(self.img_std.shape, np.nan)
        np.divide(self.img_std, lst_ave, out=out, where=lst_ave!=0)
        lght_err = out
        ref_err = self.reference_reflectance_err/self.reference_reflectance
        self.img_std = self.img_ave * np.sqrt((lght_err)**2 + (ref_err)**2)
        self.units = 'Refl. Coeffs. 1/DN/s'

class ReflectanceImage(Image):
    """Class for handling Reflectance Images, inherits Image class."""
    def __init__(self, source_image: LightImage) -> None:
        self.dir = source_image.dir
        # TODO check subject directory exists
        self.subject = source_image.subject
        self.channel = source_image.channel
        self.img_type = 'rfl'
        self.camera = source_image.camera
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
        self.polyroi = source_image.polyroi
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std

    def calibrate_reflectance(self, cali_source):
        lst_ave = self.img_ave.copy()
        self.img_ave = self.img_ave * cali_source.img_ave
        self.units = 'Reflectance'
        lght_err = self.img_std/lst_ave
        cali_err = cali_source.img_std/cali_source.img_ave
        self.img_std = self.img_ave * np.sqrt((lght_err)**2 + (cali_err)**2)

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
        # TODO check subject directory exists
        self.subject = source_image.subject
        self.channel = source_image.channel
        self.img_type = 'geo'
        self.camera = source_image.camera
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
            img = self.roi_image().astype(np.uint8)
        else:
            img = self.img_ave.astype(np.uint8)

        if method == 'ORB':
            # Initiate the ORB feature detector
            MAX_FEATURES = 500
            orb = cv2.ORB_create(MAX_FEATURES)
            points, descriptors = orb.detectAndCompute(img, self.mask)
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
            src_img = self.roi_image().astype(np.uint8)
            dest_img = self.destination.roi_image().astype(np.uint8)
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
        self.dir = source_image.dir
        # TODO check subject directory exists
        self.subject = source_image.subject
        self.channel = source_image.channel
        self.img_type = 'geo'
        self.camera = source_image.camera
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
        self.crows = chkrbrd[0]
        self.ccols = chkrbrd[1]
        self.chkrsize = chkrbrd[2]
        self.all_corners = None
        self.object_points = self.define_calibration_points()
        self.corner_points = self.find_corners()
        self.mtx, self.dist, self.rvec, self.tvec = None, None, None, None
        self.p_mtx = None
        self.calibration_error = None
        self.f_length = None

    def export_calibration(self, dir: str):
        """Print the calibration parameters and channel information to xml
        """        
        # put object properties into a dataframe        
        camera_properties = OrderedDict([
            ('Subject', self.subject),
            ('Channel', self.channel),
            ('Camera #', self.camera),
            ('Serial #', self.serial),
            ('Centre-Wavelength (nm)', self.cwl),
            ('FWHM (nm)', self.fwhm),
            ('F-Number', self.fnumber),
            ('Lens Focal Length (mm)', self.flength*1e3),
            ('Exposure Time (s)', self.exposure),
            ('Image Width (px)', self.width),
            ('Image Height (px)', self.height)            
        ])
        checkerboard_properties = OrderedDict([
            ('Checkerboard Rows', self.crows),
            ('Checkerboard Columns', self.ccols),
            ('Checkerboard Square Size (mm)', self.chkrsize*1e3)
        ])
        if self.rvec is not None:
            r_mtx, jac = cv2.Rodrigues(self.rvec)
        else:
            r_mtx = None
        camera_calibration = OrderedDict([
            ('Intrinsic Matrix', self.mtx),
            ('Rotation Matrix', r_mtx),
            ('Translation Vector', self.tvec),
            ('Projection Matrix', self.p_mtx),
            ('Distortion Coefficients', self.dist),
            ('Calibrated Focal Length (mm)', self.f_length),   
            ('Calibration Error', self.calibration_error)         
        ])
        channel_properties = OrderedDict([
            ('Camera', camera_properties),
            ('Checkerboard', checkerboard_properties),
            ('Calibration', camera_calibration)
        ])
        # write to xml
        # xml = dict2xml(channel_properties, indent="   ", iterables_repeat_wrap=True)
        xml = dict2xml.Converter(indent="    ").build(channel_properties, iterables_repeat_wrap=False)
        # save the xml file
        filename = str(Path(dir, f'{self.subject}_{self.channel}.xml'))
        with open(filename, "w") as f:
            f.write(xml)
        return xml

    def define_calibration_points(self):
        # Define calibration object points and corner locations
        objpoints = np.zeros((self.crows*self.ccols, 3), np.float32)
        objpoints[:,:2] = np.mgrid[0:self.crows, 0:self.ccols].T.reshape(-1, 2)
        objpoints *= self.chkrsize
        return objpoints

    def find_corners(self):
        # Find the chessboard corners
        gray = (self.img_ave/16).round().astype(np.uint8)
        if self.roi:
            gray = gray[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        ret, corners = cv2.findChessboardCorners(gray, (self.crows,self.ccols), flags=None)
        self.all_corners = ret

        # refine corner locations
        if self.all_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

            if self.roi:
                corners[:,:,0]+=self.roiy
                corners[:,:,1]+=self.roix
        else:
            print(f'No corners found for {self.camera} {self.cwl} nm')
            corners = None
        return corners

    def show_corners(self, ax: object=None, corner_roi: bool=False):
        # Draw and display the corners
        gray = (self.img_ave/16).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = cv2.drawChessboardCorners(rgb, (self.crows,self.ccols), self.corner_points, self.all_corners)
        if self.roi:
            img = img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        elif corner_roi and self.all_corners:
            # find the roi that bounds the corners
            self.roix = int(np.min(self.corner_points[:,:,1]))
            self.roiy = int(np.min(self.corner_points[:,:,0]))
            self.roiw = int(np.max(self.corner_points[:,:,1])-self.roix)
            self.roih = int(np.max(self.corner_points[:,:,0])-self.roiy)
            pad = int(0.3*self.roiw)
            self.roix = np.clip(self.roix-pad, 0, self.width)
            self.roiy = np.clip(self.roiy-pad, 0, self.height)            
            self.roiw = np.clip(self.roiw, 0, self.width-self.roix-2*pad)+2*pad
            self.roih = np.clip(self.roih, 0, self.height-self.roiy-2*pad)+2*pad
            img = img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]                
        ax.imshow(img, origin='upper', cmap='gray')        
        ax.set_title(f'{self.camera}: {self.cwl} nm')
        if ax is None:
            plt.show()

    def project_axes(self, ax: object=None, corner_roi: bool=False):
        # Draw and display an axis on the checkerboard
        # project a 3D axis onto the image
        axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
        axis *= self.chkrsize * 5 # default to 5 square axes
        img = (self.img_ave/16).round().astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if self.rvec is not None:
            imgpts, jac = cv2.projectPoints(axis, self.rvec, self.tvec, self.mtx, self.dist)       
            # draw the axis on the image
            corner = tuple(self.corner_points[0].ravel().astype(np.uint16))
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
        ax.set_title(f'{self.camera}: {self.cwl} nm')
        if ax is None:
            plt.show()

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
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return mtx, dist, rvecs, tvecs

class StereoPair():
    def __init__(self, source_image: LightImage, destination_image: LightImage) -> None:
        self.src = source_image
        self.dst = destination_image
        self.src_pts = None
        self.src_pt_dsc = None
        self.dst_pts = None
        self.dst_pt_dsc = None
        self.obj_pts = None
        self.matches = None
        self.f_mtx = None
        self.e_mtx = None
        self.r_mtx = None
        self.t_mtx = None
        self.stereo_err = None
        self.v_alignment = None
        self.src_r = None
        self.dst_r = None
        self.src_p = None
        self.dst_p = None
        self.q_mtx = None
        self.src_elines = None
        self.dst_elines = None
        self.stereoMatcher = None
        self.points3D = None

    def export_calibration(self, dir: str):
        """Export the stereo calibration properties to xml        

        :param dir: _description_
        :type dir: str
        """ 
        # source camera properties
        camera_A_properties = OrderedDict([
            ('Subject', self.src.subject),
            ('Channel', self.src.channel),
            ('Camera #', self.src.camera),
            ('Serial #', self.src.serial),
            ('Centre-Wavelength (nm)', self.src.cwl),
            ('FWHM (nm)', self.src.fwhm),
            ('F-Number', self.src.fnumber),
            ('Lens Focal Length (mm)', self.src.flength*1e3),
            ('Exposure Time (s)', self.src.exposure),
            ('Image Width (px)', self.src.width),
            ('Image Height (px)', self.src.height)            
        ])
        camera_B_properties = OrderedDict([
            ('Subject', self.dst.subject),
            ('Channel', self.dst.channel),
            ('Camera #', self.dst.camera),
            ('Serial #', self.dst.serial),
            ('Centre-Wavelength (nm)', self.dst.cwl),
            ('FWHM (nm)', self.dst.fwhm),
            ('F-Number', self.dst.fnumber),
            ('Lens Focal Length (mm)', self.dst.flength*1e3),
            ('Exposure Time (s)', self.dst.exposure),
            ('Image Width (px)', self.dst.width),
            ('Image Height (px)', self.dst.height)
        ])
        # stereo properties
        stereo_properties = OrderedDict([
            ('Fundamental Matrix', self.f_mtx),
            ('Essential Matrix', self.e_mtx),
            ('Rotation Matrix', self.r_mtx),
            ('Translation Vector', self.t_mtx),
            ('Stereo Calibration Error', self.stereo_err),
            ('Rectification Matrix', self.q_mtx)
        ])
        channel_properties = OrderedDict([
            ('Camera A', camera_A_properties),
            ('Camera B', camera_B_properties),
            ('Stereo Calibration', stereo_properties)
        ])
        # write to xml       
        # xml = dict2xml(channel_properties, indent="   ", iterables_repeat_wrap=True)
        xml = dict2xml.Converter(indent="    ").build(channel_properties, iterables_repeat_wrap=False)
        # save the xml file
        filename = str(Path(dir, f'A{self.src.channel}_B{self.dst.channel}.xml'))
        with open(filename, "w") as f:
            f.write(xml)
        return xml

    def calibrate_stereo(self):
        """Calibrate the given stereo pair of cameras
        """        
        if self.obj_pts == None:
            self.obj_pts = [self.src.object_points]
        if self.src_pts == None:
            self.src_pts = [self.src.corner_points]
        if self.dst_pts == None:
            self.dst_pts = [self.dst.corner_points]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001) 
        try:
            ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
                self.obj_pts, 
                self.src_pts, self.dst_pts, 
                self.src.mtx, self.src.dist,
                self.dst.mtx, self.dst.dist, 
                (self.src.width, self.src.height), 
                criteria = criteria, 
                flags = cv2.CALIB_FIX_INTRINSIC)
        except:
            print('stop')
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

    def compute3D(self):
        """Compute 3D point locations
        """        
        points4D = cv2.triangulatePoints(self.src.p_mtx, self.dst.p_mtx, self.src.corner_points, self.dst.corner_points)
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
        color = channel_cols(self.dst.camera) # tuple(np.random.randint(0,255,3).tolist())
        ax.scatter3D(x_pts, y_pts, z_pts, color=color)
        # ax.set_xlim(-self.src.chkrsize*self.src.ccols, +self.src.chkrsize*self.src.ccols)
        # ax.set_ylim(-self.src.chkrsize*self.src.ccols, +self.src.chkrsize*self.src.ccols)
        # ax.set_zlim(-self.src.chkrsize*self.src.ccols, +self.src.chkrsize*self.src.ccols)
        title = f'{self.dst.camera}: {self.dst.cwl} nm'
        ax.set_title(title)
        # TODO show the plot if the ax was none

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
            img = (self.dst.img_ave.round()/16).astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            pts = self.dst_pts
            lines = self.dst_elines
            title = f'{self.dst.camera}: {self.dst.cwl} nm'
        elif view == 'src':
            r,c = self.src.img_ave.shape
            if ax.title.get_text() == '':
                img = (self.src.img_ave.round()/16).astype(np.uint8)
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
            img = cv2.line(img, (x0,y0), (x1,y1), color,1)
            img = cv2.circle(img,tuple(pt.flatten().astype(int)),5,color,-1)            
        
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(img, origin='upper')
            ax.set_title(title)
            fig.show()
        else:
            ax.imshow(img, origin='upper')
            ax.set_title(title)

    def rectify(self,ax: object=None) -> None:
        """Rectify the source and destination images
        """        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.src.mtx, self.src.dist,
                self.dst.mtx, self.dst.dist,
                (self.src.height, self.src.width),
                self.r_mtx, self.t_mtx, alpha=-1, newImageSize=(0,0))
        self.src_r = R1
        self.dst_r = R2
        self.src_p = P1
        self.dst_p = P2
        self.q_mtx = Q

        self.v_alignment = self.dst_p[0,3] == 0.0            
        
        self.src_map1, self.src_map2 = cv2.initUndistortRectifyMap(
            self.src.mtx, self.src.dist, self.src_r, self.src_p, 
            (self.src.height, self.src.width), cv2.CV_32FC1)
        
        self.dst_map1, self.dst_map2 = cv2.initUndistortRectifyMap(
            self.dst.mtx, self.dst.dist, self.dst_r, self.dst_p, 
            (self.dst.height, self.dst.width), cv2.CV_32FC1)
        
        src_img = (self.src.img_ave.round()/16).astype(np.uint8)
        src_img = cv2.cvtColor(src_img,cv2.COLOR_GRAY2BGR)
        dst_img = (self.dst.img_ave.round()/16).astype(np.uint8)
        dst_img = cv2.cvtColor(dst_img,cv2.COLOR_GRAY2BGR)        

        src_img = cv2.remap(src_img, self.src_map1, self.src_map2, cv2.INTER_LINEAR)
        dst_img = cv2.remap(dst_img, self.dst_map1, self.dst_map2, cv2.INTER_LINEAR)
        
        if self.v_alignment:
            src_img = cv2.rotate(src_img, cv2.ROTATE_90_CLOCKWISE)
            dst_img = cv2.rotate(dst_img, cv2.ROTATE_90_CLOCKWISE)        
            
        self.src_rect = src_img
        self.dst_rect = dst_img

        # src_img = cv2.rectangle(self.src_rect, (roi1[0],roi1[1]), (roi1[0]+roi1[2],roi1[1]+roi1[3]), (0,0,255), 2)
        # dst_img = cv2.rectangle(self.dst_rect, (roi2[0],roi2[1]), (roi2[0]+roi2[2],roi2[1]+roi2[3]), (0,0,255), 2)

        title = f'{self.dst.camera}: {self.dst.cwl} nm'

        fig2, ax2 = plt.subplots(1,2)

        ax2[0].imshow(dst_img, origin='upper')  
        ax2[1].imshow(src_img, origin='upper')
        ax2[0].set_title(self.dst.channel)
        ax2[1].set_title(self.src.channel+' Stereo Err.: '+str(self.stereo_err))

        h_lines = np.arange(0, self.src_rect.shape[0], self.src_rect.shape[0]/40) 
        for h_line in h_lines:
            ax2[0].axhline(y = h_line, color = 'r', linestyle = '-', lw=0.4)      
            ax2[1].axhline(y = h_line, color = 'r', linestyle = '-', lw=0.4)      
        fig2.show()

        if self.dst_elines is not None:
            self.draw_epilines('dst')

        if self.src.camera == self.dst.camera:
            ax.imshow(src_img, origin='upper')    
            ax.set_title(title)
        else:
            ax.imshow(dst_img, origin='upper')    
            ax.set_title(title)

    def compute_disparity(self, ax: object=None) -> None:
        """Compute a disparity map for the given image pair.
        """        
        matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.stereoMatcher = matcher      

        src_img = cv2.cvtColor(self.src_rect,cv2.COLOR_BGR2GRAY)
        dst_img = cv2.cvtColor(self.dst_rect,cv2.COLOR_BGR2GRAY)  

        self.disparity = matcher.compute(src_img, dst_img)
        if ax is not None:
            ax.imshow(self.disparity, origin='upper', cmap='gray')
            ax.set_title(f'{self.dst.camera}: {self.dst.cwl} nm')

    def depth_from_disparity(self, ax: object=None) -> None:
        """Compute the 3D point cloud from the disparity map, Q matrix and 
        rectified projection matrices.
        """
        self.points3D = cv2.reprojectImageTo3D(self.disparity, self.q_mtx)        
            
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
            self.src_pts, self.src_pt_dsc = self.find_features('source')
            self.dst_pts, self.dst_pt_dsc = self.find_features('destination')        
        F, mask = cv2.findFundamentalMat(self.src_pts,self.dst_pts,cv2.FM_LMEDS)
        self.f_mtx = F
        # update the matches to only include those that are inliers
        self.src_pts = self.src_pts[mask.ravel()==1]
        self.dst_pts = self.dst_pts[mask.ravel()==1]
        return F
    
    def find_essential_mtx(self) -> np.ndarray:
            """Find the essential matrix between the source and destination
            cameras

            :return: Essential matrix
            :rtype: np.ndarray
            """                
            self.src_pts = self.src.corner_points
            self.dst_pts = self.dst.corner_points
       
            E, mask = cv2.findEssentialMat(
                self.src_pts,
                self.dst_pts,
                self.src.mtx,
                self.src.dist,
                self.dst.mtx,
                self.dst.dist              
                )
            self.e_mtx = E
            # update the matches to only include those that are inliers
            self.src_pts = self.src_pts[mask.ravel()==1]
            self.dst_pts = self.dst_pts[mask.ravel()==1]
            return E

    def find_matches(self, use_corners: bool=False) -> int:
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
            self.src_pts, self.src_pt_dsc = self.find_features('source')
            self.dst_pts, self.dst_pt_dsc = self.find_features('destination')
        # find matches
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        GOOD_MATCH_PERCENT = 0.10
        matches = list(matcher.match(self.src_pt_dsc, self.dst_pt_dsc, None))
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]    
        self.matches = matches
        return len(matches)    

    def find_features(self, view: str) -> tuple:
        """Find feature points and descriptors in the source or destination image

        :param view: 'source' or 'destination' iamge to perform search over
        :type view: str
        :return: feature points and descriptors
        :rtype: tuple
        """        
        if view == 'source':
            img = self.src.img_ave
        elif view == 'destination':
            img = self.dst.img_ave
        else:
            raise ValueError('View must be "source" or "destination"')
                
        img = (img.round()/16).astype(np.uint8) # convert image to 8-bit
        MAX_FEATURES = 500
        orb = cv2.ORB_create(MAX_FEATURES)
        points, descriptors = orb.detectAndCompute(img, None) # TODO allow for an actual mask to be applied, rather than just None
        return points, descriptors

    def draw_matches(self, ax: object=None) -> None:
        """Draw the matching points in the source and destination images
        """       
        src_img = (self.src.img_ave.round()/16).astype(np.uint8)
        dst_img = (self.dst.img_ave.round()/16).astype(np.uint8)
        img = cv2.drawMatches(src_img, self.src_pts, dst_img, self.dst_pts, self.matches, None)
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

    def compute_depth(self) -> None:
        """Compute the depth of each pixel in the source and destination images
        """        
        pass

def grid_plot(title: str=None, projection: str=None):
    cam_ax = {}
    if projection == '3d':
        fig, ax = plt.subplots(3,3, figsize=(FIG_W,FIG_W), subplot_kw=dict(projection='3d'))
    else:
        fig, ax = plt.subplots(3,3, figsize=(FIG_W,FIG_W))
    # TODO update this according to camera number
    cam_ax[2] = ax[0][0] # 400
    cam_ax[5] = ax[0][1] # 950
    cam_ax[7] = ax[0][2] # 550
    cam_ax[4] = ax[1][0] # 735
    cam_ax[8] = ax[1][1] # Histogram
    cam_ax[0] = ax[1][2] # 850
    cam_ax[6] = ax[2][0] # 650
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

def load_reflectance_calibration(subject: str='reflectance_calibration', roi: bool=False, caption: str=None) -> Dict:
    channels = sorted(list(Path('..', 'data', subject).glob('[!.]*')))
    cali_imgs = {} # store the calibration objects in a dictionary
    title = 'Spectralon 99% Calibration Target'
    fig, ax = grid_plot(title)
    for channel_path in channels:
        channel = channel_path.name
        # load the calibration target images
        cali = LightImage(subject, channel)
        cali.image_load()
        # load the calibration target dark frames
        dark_cali = DarkImage(subject, channel)
        dark_cali.image_load()
        # subtract the dark frame
        cali.dark_subtract(dark_cali)
        # show
        cali.image_display(roi=roi, ax=ax[cali.camera], histo_ax=ax[8])
        cali_imgs[channel] = cali
    show_grid(fig, ax)
    if caption is not None:
        grid_caption(caption)
    return cali_imgs

def calibrate_reflectance(cali_imgs: Dict, caption: Tuple[str, str]=None, clip: float=0.25) -> Dict:
    """Calibrated the reflactance correction coefficients for images
    of the Spectralon reflectance target.

    :param subject: directory of target images, defaults to 'reflectance_calibration'
    :type subject: str, optional
    :param clip: clip the image histogram to the given percentile, defaults to None
    :type clip: float, optional
    :return: Dictionary of reflectance correction coefficient CalibrationImages
    :rtype: Dict
    """
    channels = list(cali_imgs.keys())
    cali_coeffs = {}
    title = 'Reflectance Calibration Coefficients'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption[0])
    title = 'Reflectance Calibration Coefficients Error'
    fig1, ax1 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[1])
    for channel in channels:
        # load the calibration target images
        cali = cali_imgs[channel]
        print(f'Finding Reflectance Correction for: {cali.camera} ({int(cali.cwl)} nm)')
        # apply exposure correction
        cali.correct_exposure()
        # compute calibration coefficient image
        cali_coeff = CalibrationImage(cali)
        cali_coeff.mask_target(clip)
        cali_coeff.compute_reflectance_coefficients()
        cali_coeff.image_display(roi=True, ax=ax[cali_coeff.camera], histo_ax=ax[8])
        cali_coeff.image_display(roi=True, noise=True, ax=ax1[cali_coeff.camera], histo_ax=ax1[8])
        cali_coeffs[channel] = cali_coeff
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return cali_coeffs

def load_sample(subject: str='sample', dark_subject: str='sample', roi: bool=False, caption: str=None, threshold: Tuple[float,float]=(None,None)) -> Dict:
    """Load images of the sample.

    :param subject: Directory of sample images, defaults to 'sample'
    :type subject: str, optional
    :return: Dictionary of sample LightImages (units of DN)
    :rtype: Dict
    """
    channels = sorted(list(Path('..', 'data', subject).glob('[!.]*')))
    smpl_imgs = {} # store the calibration objects in a dictionary
    title = f'{subject} Images'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption[0])
    title = f'{subject} Images SNR'
    fig1, ax1 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[1])
    for channel_path in channels:
        channel = channel_path.name
        smpl = LightImage(subject, channel)
        smpl.image_load()
        print(f'Loading {subject}: {smpl.camera} ({int(smpl.cwl)} nm)')
        dark_smpl = DarkImage(dark_subject, channel)
        dark_smpl.image_load()
        # Check exposure times are equal
        light_exp = smpl.exposure
        dark_exp = dark_smpl.exposure
        if light_exp != dark_exp:
            raise ValueError(f'Light and Dark Exposure Times are not equal: {light_exp} != {dark_exp}')
        # subtract the dark frame
        smpl.dark_subtract(dark_smpl)
        # show
        smpl.image_display(roi=roi, ax=ax[smpl.camera], histo_ax=ax[8], threshold=threshold[0])
        smpl.image_display(roi=roi, snr=True, ax=ax1[smpl.camera], histo_ax=ax1[8], threshold=threshold[1])
        smpl_imgs[channel] = smpl
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return smpl_imgs

def load_dark_frames(subject: str='sample', roi: bool=False, caption: Tuple[str, str]=None) -> Dict:
    channels = sorted(list(Path('..', 'data', subject).glob('[!.]*')))
    dark_imgs = {} # store the calibration objects in a dictionary
    title = f'{subject} Mean Dark Images'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption[0])
    title = f'{subject} Std. Dev. Dark Images'
    fig1, ax1 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[1])
    for channel_path in channels:
        channel = channel_path.name
        dark_smpl = DarkImage(subject, channel)
        dark_smpl.image_load()
        print(f'Loading {subject}: {dark_smpl.camera} ({int(dark_smpl.cwl)} nm)')
        # show
        dark_smpl.image_display(roi=roi, ax=ax[dark_smpl.camera], histo_ax=ax[8])
        dark_smpl.image_display(noise=True, roi=roi, ax=ax1[dark_smpl.camera], histo_ax=ax1[8])
        dark_imgs[channel] = dark_smpl
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    return dark_imgs

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

def get_reference_reflectance(cali_coeffs: Dict) -> pd.DataFrame:
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
    reference = pd.DataFrame({'cwl': cwl, 'reference': ref_val, 'error': ref_err}, index=channels)
    reference.sort_values('cwl', inplace=True)
    return reference

def apply_reflectance_calibration(smpl_imgs: Dict, cali_coeffs: Dict, caption: Tuple[str,str,str]=None) -> Dict:
    """Apply reflectance calibration coefficients to the sample images.

    :param sample_imgs: Dictionary of LightImage objects (units of DN)
    :type sample_imgs: Dict
    :return: Dictionary of Reflectance Images (units of Reflectance)
    :rtype: Dict
    """
    channels = list(smpl_imgs.keys())
    reflectance = {}
    title = 'Sample Reflectance'
    fig, ax = grid_plot(title)
    if caption is not None:
        grid_caption(caption[0])
    title = 'Sample Reflectance Noise'
    fig1, ax1 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[1])
    title = 'Sample Reflectance SNR'
    fig2, ax2 = grid_plot(title)
    if caption is not None:
        grid_caption(caption[2])
    vmax=0.0
    for channel in channels:
        smpl = smpl_imgs[channel]
        # apply exposure correction
        smpl.correct_exposure()
        # apply calibration coefficients
        cali_coeff = cali_coeffs[channel]
        refl = ReflectanceImage(smpl)
        refl.calibrate_reflectance(cali_coeff)
        refl_max = np.max(refl.img_ave[np.isfinite(refl.img_ave)])
        if refl_max > vmax:
            vmax = refl_max
        # save the reflectance image
        refl.save_tiff()
        reflectance[channel] = refl
    for channel in channels:
        refl = reflectance[channel]
        refl.image_display(roi=True, ax=ax[refl.camera], histo_ax=ax[8], vmin=0.0, vmax=vmax)
        refl.image_display(roi=True, noise=True, ax=ax1[refl.camera], histo_ax=ax1[8])
        refl.image_display(roi=True, snr=True, ax=ax2[refl.camera], histo_ax=ax2[8], vmin=0.0, vmax=300)
    show_grid(fig, ax)
    show_grid(fig1, ax1)
    show_grid(fig2, ax2)
    return reflectance

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

def plot_roi_reflectance(
        refl_imgs: Dict,
        reference_reflectance: pd.DataFrame=None,
        show_error_limit: bool=False,
        weighted: bool=False,
        caption: str=None) -> pd.DataFrame:
    """Plot the reflectance over the Region of Interest

    :param refl_imgs: Dictionary of ReflectanceImage objects
    :type refl_imgs: Dict
    :return: Pandas DataFrame of reflectance over the ROI
    :rtype: pd.DataFrame
    """
    cwls = []
    means = []
    stds = []
    errs = []
    wt_means = []
    wt_stds = []
    channels = list(refl_imgs.keys())
    for channel in channels:
        refl_img = refl_imgs[channel]
        mean, std, err, wt_mean, wt_std = refl_img.image_stats(polyroi=True)
        cwl = refl_img.cwl
        cwls.append(cwl)
        means.append(mean)
        stds.append(std)
        errs.append(err)
        wt_means.append(wt_mean)
        wt_stds.append(wt_std)
    results = pd.DataFrame({'cwl':cwls, 'reflectance':means, 'standard deviation':stds, 'err':errs, 'reflectance (wt)':wt_means, 'std (wt)':wt_stds})
    results.sort_values(by='cwl', inplace=True)
    # results.plot(x='cwl', y='mean', yerr='std')

    fig = plt.figure()
    plt.grid(visible=True)
    if weighted:
        plt.errorbar(
                x=results.cwl,
                y=results['reflectance (wt)'],
                yerr=results['std (wt)'],
                fmt='o-',
                ecolor='k',
                capsize=2.0)
    else:
        plt.errorbar(
                x=results.cwl,
                y=results.reflectance,
                yerr=results['standard deviation'],
                fmt='o-',
                # ecolor='k',
                capsize=2.0)
        if show_error_limit:
            plt.errorbar(
                    x=results.cwl,
                    y=results.reflectance,
                    yerr=results.err,
                    fmt='',
                    linestyle='',
                    ecolor='r',
                    capsize=6.0)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    # plt.title('Sample Reflectance Mean ± 1σ over ROI')

    if reference_reflectance is not None:
        plt.errorbar(
            x=reference_reflectance.cwl,
            y=reference_reflectance.reflectance,
            yerr=reference_reflectance['standard deviation'],
            fmt='o-',
            capsize=2.0
        )
        if show_error_limit:
            plt.errorbar(
                    x=results.cwl,
                    y=results.reflectance,
                    yerr=results.err,
                    fmt='x-',
                    ecolor='r',
                    capsize=5.0)

    if caption is not None:
        grid_caption(caption)

    results['SNR Limit'] = results['reflectance'] / results['err']
    results['SNR'] = results['reflectance'] / results['standard deviation']

    return results

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
            dark_path.symlink_to(src_dark_path.resolve(), target_is_directory=True)      
    else:
        raise ValueError(f'Dark frame directory does not exist: {src_dark_path}')
    # build the raw data directory
    raw_path = Path(scene_path, 'raw')
    if src_scene_path.exists():
        if not raw_path.exists():
            raw_path.symlink_to(src_scene_path.resolve(), target_is_directory=True)
    else:
        raise ValueError(f'Scene directory does not exist: {src_scene_path}')
    # build the products directory
    prod_path = Path(scene_path, 'products')
    prod_path.mkdir(exist_ok=True)

    return raw_path, dark_path

def load_geometric_calibration(scene_path: Path, dark_path: Path, caption: str=None) -> Dict:
    """Load the geometric calibration images.

    :param subject: directory of target images, defaults to 'geometric_calibration'
    :type subject: str, optional
    :return: Dictionary of geometric correction LightImages
    :rtype: Dict
    """    
    channels = sorted(list(scene_path.glob('[!._]*')))
    geocs = {}
    fig, ax = grid_plot('Geometric Calibration Target')
    if caption is not None:
        grid_caption(caption)
    for channel_path in channels:
        channel = channel_path.name
        # load the geometric calibration images
        geoc = LightImage(scene_path, channel)
        geoc.image_load()
        print(f'Loading Geometric Target for: {geoc.camera} ({geoc.cwl} nm)')
        # load the geometric calibration dark frames
        dark_geoc = DarkImage(dark_path, channel)
        dark_geoc.image_load()
        # subtract the dark frame
        geoc.dark_subtract(dark_geoc)
        # show
        geoc.image_display(roi=False, ax=ax[geoc.camera], histo_ax=ax[8])
        geocs[channel] = geoc
    show_grid(fig, ax)
    return geocs

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
        src.show_corners(ax=ax[src.camera], corner_roi=True)
        corners[channel] = src
    show_grid(fig, ax)
    return corners

def calibrate_homography(geocs: Dict, caption: Tuple[str, str]=None) -> Dict:
    """Calibrate the homography matrix for images of the geometric calibration
    target.

    :param subject: directory of target images, defaults to 'geometric_calibration'
    :type subject: str, optional
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
