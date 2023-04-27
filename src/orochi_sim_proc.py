"""A library of functions for processing OROCHI Simulator Images.

Roger Stabbins
Rikkyo University
21/04/2023
"""
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from typing import Tuple
import orochi_sim_ctrl as osc

class Image:
    """Super class for handling image import and export.
    """
    def __init__(self, subject: str, channel: str, img_type: str) -> None:
        """Initialise properties. Most of these are populated during image load.
        """
        self.dir = Path('..', 'data', subject, channel)
        # TODO check subject directory exists
        self.subject = subject
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
        self.units = ''
        self.n_imgs = None
        self.img_stk = None
        self.img_ave = None
        self.img_std = None

    def image_load(self):
        """Load images from the subject directory for the given type,
        populate properties, and compute averages and standard deviation.
        """
        # get list of images of given type in the subject directory
        files = list(self.dir.glob('*'+self.img_type+'.tif'))
        # set n_imgs
        self.n_imgs = len(files)
        self.units = 'Raw DN'
        img_list = []
        if files == []:
            raise FileNotFoundError(f'Error: no {self.img_type} images found in {self.dir}')
        for f, file in enumerate(files):
            img = tiff.TiffFile(file)
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

        self.img_stk = np.dstack(img_list)
        # average stack
        self.img_ave = np.mean(self.img_stk, axis=2)
        # std. stack
        self.img_std = np.std(self.img_stk, axis=2)

    def roi_image(self) -> np.ndarray:
        """Returns the region of interest of the image.
        """
        return self.img_ave[self.roix:self.roix+self.roiw,
                            self.roiy:self.roiy+self.roih]

    def roi_std(self) -> np.ndarray:
        """Returns the region of interest of the error image.
        """
        return self.img_std[self.roix:self.roix+self.roiw,
                            self.roiy:self.roiy+self.roih]

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

        self.img_stk = self.img_stk / self.exposure
        self.img_ave = self.img_ave / self.exposure
        self.img_std = self.img_std / self.exposure # assume exposure err. negl.
        self.units = 'DN/s'

    def image_stats(self) -> None:
        """Print image statistics
        """
        img_ave_mean = np.mean(self.img_ave)
        print(f'Average Image Mean: {img_ave_mean} {self.units}')
        img_std_mean = np.mean(self.img_std)
        print(f'Noise Image Mean: {img_std_mean} {self.units}')

    def image_display(self, roi: bool=False) -> None:
        """Display the image mean and standard deviation in one frame.
        """
        if self.img_type == 'rfl':
            vmin = 0.0
            vmax = 1.0
        else:
            vmin = None
            vmax = None
        # set the size of the window for the two images
        fig, ax = plt.subplots(
            nrows=2, ncols=1, sharey=True, sharex=True,figsize=(5.5, 5.8))
        fig.suptitle(f'subject: {self.subject}, Channel: {self.channel}')
        # put the mean frame in
        if roi:
            img = self.roi_image()
        else:
            img = self.img_ave
        ave = ax[0].imshow(img, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(ave, ax=ax[0], label=self.units)
        ax[0].set_title('Image Average')
        # put the std frame in
        if roi:
            img_std = self.roi_std()
        else:
            img_std = self.img_std
        std = ax[1].imshow(img_std, origin='lower')
        ax[1].set_title('Image Std. Dev.')
        plt.colorbar(std, ax=ax[1], label=self.units)
        # show
        plt.tight_layout()
        plt.show()
        # TODO add histograms

    def save_tiff(self, save_stack: bool=False):
        """Save the average and error images to TIF files"""
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
        out_img = self.img_ave.astype(np.float32)
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
    def __init__(self, subject: str, channel: str) -> None:
        Image.__init__(self,subject, channel, 'drk')
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
    def __init__(self, subject: str, channel: str) -> None:
        Image.__init__(self,subject, channel, 'img')

    def dark_subtract(self, dark_image: DarkImage) -> None:
        lst_ave = self.img_ave.copy()
        self.img_stk -= dark_image.img_stk
        self.img_ave -= dark_image.img_ave
        # quadrture sum the image noise with the dark signal noise
        lght_err = self.img_std/lst_ave
        drk_err = dark_image.img_std/dark_image.img_ave
        self.img_std = self.img_ave * np.sqrt((lght_err)**2 + (drk_err)**2)
        self.units = 'Above-Bias Signal DN'


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
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_stk = source_image.img_stk
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

    def compute_reflectance_coefficients(self):
        """Compute the reflectance coefficients for each pixel of the
        calibration target.
        """
        lst_ave = self.img_ave.copy()
        self.img_stk = self.reference_reflectance / self.img_stk
        self.img_ave = self.reference_reflectance / self.img_ave
        lght_err = self.img_std/lst_ave
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
        self.units = source_image.units
        self.n_imgs = source_image.n_imgs
        self.img_stk = source_image.img_stk
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std

    def calibrate_reflectance(self, cali_source):
        lst_ave = self.img_ave.copy()
        self.img_ave = self.img_ave * cali_source.img_ave
        self.units = 'Reflectance'
        lght_err = self.img_std/lst_ave
        cali_err = cali_source.img_std/cali_source.img_ave
        self.img_std = self.img_ave * np.sqrt((lght_err)**2 + (cali_err)**2)

class CoAlignedImage(Image):
    def __init__(self,
                    source_image: LightImage,
                    destination_image: LightImage,
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
        self.img_stk = source_image.img_stk
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.roi = roi
        self.destination = destination_image
        self.homography = homography
        self.matches = None

    def find_homography(self, method: str):

        if self.roi:
            query_img = self.roi_image().astype(np.int8)
            train_img = self.destination.roi_image().astype(np.int8)
        else:
            query_img = self.img_ave.astype(np.uint8)
            train_img = self.destination.img_ave.astype(np.uint8)

        # Initiate the ORB feature detector
        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.20

        orb = cv2.ORB_create(MAX_FEATURES)

        # Find features
        keypoints1, descriptors1 = orb.detectAndCompute(query_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(train_img, None)

        # Find feature matches
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(descriptors1, descriptors2, None))
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw the feature matches
        imMatches = cv2.drawMatches(
                        query_img, keypoints1,
                        train_img, keypoints2,
                        matches, None)

        # Get feature coordinates in each image
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find the homography matrix
        homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        self.homography = homography
        self.matches = imMatches

    def align_images(self):

        # apply the transform
        if self.roi:
            query_img = self.roi_image().astype(np.int8)
        else:
            query_img = self.img_ave.astype(np.uint8)
        height, width = query_img.shape
        hmgr = self.homography
        query_reg = cv2.warpPerspective(query_img, hmgr, (width, height))
        if self.roi:
            train_img = self.destination.roi_image().astype(np.int8)
        else:
            train_img = self.destination.img_ave.astype(np.uint8)
        fig = plt.figure()
        plt.imshow(query_reg-train_img, origin='lower')
        plt.show()

class GeoCalImage(Image):
    def __init__(self, source_image: LightImage, roi: bool=False) -> None:
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
        self.img_stk = source_image.img_stk
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.roix = source_image.roix
        self.roiy = source_image.roiy
        self.roiw = source_image.roiw
        self.roih = source_image.roih
        self.roi = roi
        self.crows = 4
        self.ccols = 3
        self.chkrsize = 5.0E-3
        self.all_corners = None
        self.object_points = self.define_calibration_points()
        self.corner_points = self.find_corners()
        self.mtx, self.dist, self.rvecs, self.tvecs = None, None, None, None

    def define_calibration_points(self):
        # Define calibration object points and corner locations
        objpoints = np.zeros((self.crows*self.ccols, 3), np.float32)
        objpoints[:,:2] = np.mgrid[0:self.crows, 0:self.ccols].T.reshape(-1, 2)
        objpoints *= self.chkrsize
        return objpoints

    def find_corners(self):
        # Find the chessboard corners
        gray = self.img_ave.astype(np.uint8)
        if self.roi:
            gray = gray[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        ret, corners = cv2.findChessboardCorners(gray, (self.crows,self.ccols))
        self.all_corners = ret

        # refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        corners[:,:,0]+=self.roiy
        corners[:,:,1]+=self.roix
        return corners

    def show_corners(self):
        # Draw and display the corners
        gray = self.img_ave.astype(np.uint8)
        img = cv2.drawChessboardCorners(gray, (self.crows,self.ccols), self.corner_points, self.all_corners)
        if self.roi:
            img = img[self.roix:self.roix+self.roiw, self.roiy:self.roiy+self.roih]
        plt.imshow(img, origin='lower')
        plt.show()

    def calibrate_camera(self):
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([self.object_points], [self.corner_points], (self.width, self.height), None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return mtx, dist, rvecs, tvecs

class AlignedImage(Image):
    def __init__(self, source_image: LightImage, source_geocal: GeoCalImage, destination_geocal: GeoCalImage) -> None:
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
        self.img_stk = source_image.img_stk
        self.img_ave = source_image.img_ave
        self.img_std = source_image.img_std
        self.source_geocal = source_geocal
        self.destination_geocal = destination_geocal
        self.source_R, self.source_P, self.dest_R, self.dest_P = None, None, None, None

    def undistort(self):
        # undistort the image
        h, w = self.img_ave.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.source_geocal.mtx, self.source_geocal.dist, (w,h), 1, (w,h))
        dst = cv2.undistort(self.img_ave, self.source_geocal.mtx, self.source_geocal.dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        self.img_ave = dst
        return dst

    def stereo_calibrate(self):
        # Stereo calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate([self.source_geocal.object_points], [self.source_geocal.corner_points], [self.destination_geocal.corner_points], self.source_geocal.mtx, self.source_geocal.dist, self.destination_geocal.mtx, self.destination_geocal.dist, (self.width, self.height), criteria=criteria, flags=flags)
        # compute rectification parameters
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(self.source_geocal.mtx, self.source_geocal.dist, self.destination_geocal.mtx, self.destination_geocal.dist, (self.width, self.height), R, T, alpha=0)
        return R1, P1, R2, P2

    def coalign(self):
        # co-align images
        map1, map2 = cv2.initUndistortRectifyMap(self.source_geocal.mtx, self.source_geocal.dist, self.source_R, self.source_P, (self.width, self.height), cv2.CV_16SC2)
        gray = self.img_ave.astype(np.uint8)
        img_ave = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)
        return img_ave
