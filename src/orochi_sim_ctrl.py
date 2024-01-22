"""A library of functions for controlling the OROCHI Simulator.

Roger Stabbins
Rikkyo University
21/04/2023
"""

import ctypes
from datetime import date
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import scipy.optimize as opt
import tifffile as tiff
import tisgrabber as tis

# Log of default exposures for spectralon, sample, and checkerboard imaging

INIT_EXPOSURES = {
    'SPECTRALON': {
        0: 0.027141,
        1: 0.015685,
        2: 0.067584,
        3: 0.005559,
        4: 0.003785,
        5: 0.194560,
        6: 0.004960,
        7: 0.010865,
    },
    'SAMPLE': {
        0: 0.466847,
        1: 0.319456,
        2: 1.429672,
        3: 0.113773,
        4: 0.070943,
        5: 3.205451,
        6: 0.099055,
        7: 0.206448,
    },
    'SAMPLE-FIXEDLAMP': {
        0: 0.538227,
        1: 0.315676,
        2: 1.237319,
        3: 0.114834,
        4: 0.072598,
        5: 3.254072,
        6: 0.093471,
        7: 0.230196,
    },
    'CHECKERBOARD': {
        0: 0.448433,
        1: 0.317814,
        2: 1.370842,
        3: 0.091816,
        4: 0.054621,
        5: 2.729796,
        6: 0.091901,
        7: 0.171748
    }
}

class Channel:
    """Class for controlling a single camera channel.
    """
    def __init__(self, name, camera_props, ic):
        self.name = name
        if camera_props is not None:
            self.number = camera_props['number']
            self.camera_props = camera_props
        else:
            self.number = None
            self.camera_props = None
        self.ic = ic
        self.grabber = ic.IC_CreateGrabber()
        self.connect()
        self.init_camera_stream()
        self.width, self.height, self.buffer_size, self.bpp = self.get_image_info()
        self.max_dn = None
        self.session = None
        self.scene = None

    def connect(self):
        """Connect the camera to the grabber.
        """
        self.ic.IC_OpenDevByUniqueName(self.grabber, tis.T(self.name))

        # frameReadyCallbackfunc = self.ic.FRAMEREADYCALLBACK(frameReadyCallback)
        # userdata = CallbackUserdata()
        # devicelostcallbackfunc = self.ic.DEVICELOSTCALLBACK(deviceLostCallback)

        # userdata.devicename = f'{self.number} ({self.name})'
        # userdata.connected = True

        # self.ic.IC_SetCallbacks(self.grabber,
        #                 frameReadyCallbackfunc, None,
        #                 devicelostcallbackfunc, userdata)

        # check the device is connected
        if self.ic.IC_IsDevValid(self.grabber):
            print(f'Device {self.number} ({self.name}) succesfully connected.')
        else:
            err_string = f'Camera {self.number} ({self.name}) did not connect'
            self.ic.IC_MsgBox( tis.T(err_string),tis.T("Connection Error"))

    def init_camera_stream(self) -> None:
        """Initialise the camera and check the live video stream.
        """
        self.ic.IC_StartLive(self.grabber, 0)
        # self.ic.IC_MsgBox(tis.T("Camera Stream Check: Click OK to stop"), tis.T("Initialising Camera Feed"))
        self.ic.IC_StopLive(self.grabber)

    def set_defaults(self, 
                     bit_depth = 12,
                     fps=30.0,
                     black_level=4,
                     gain=4.27,
                     exposure=1.0/100, 
                     auto_exposure=1):
        """Set default properties for each camera.

        :param exposure: exposure time (seconds), defaults to 1.0/100
        :type exposure: float, optional
        :param auto_exposure: AE activation flag, defaults to 1
        :type auto_exposure: int, optional
        :param black_level: black offset level, in 8-bit DN range, defaults to
            26 (10% of the range, for linearity)
        :type black_level: int, optional
        """
        print('***********************************')
        print(f'Setting Properties for Camera {self.number} ({self.name})')
        print('***********************************')
        # 8-bit
        if bit_depth == 8:
            vid_format = "Y800 (1920x1200)"
            sink_format_id = 0
        # 12-bit
        if bit_depth == 12:
            vid_format = "Y16 (1920x1200)"
            sink_format_id = 4
        ret = self.ic.IC_SetVideoFormat(self.grabber, tis.T(vid_format))
        if ret != 1:
            print(f'Video Format error code: {ret}')
        print(f'Video Format set to : {vid_format}')
        ret = self.ic.IC_SetFormat(self.grabber, ctypes.c_int(sink_format_id))
        if ret != 1:
            print(f'Sink Format error code: {ret}')
        print(f'Sink Format set to : "{tis.SinkFormats(sink_format_id)}"')        
        # ret = self.ic.IC_SetFrameRate(self.grabber, ctypes.c_float(fps))
        # print(f'Frame Rate set to : {fps} FPS')
        ret = self.set_frame_rate(fps)
        self.init_camera_stream()
        self.get_image_info()

        # If exposure keyword set, look up from INIT_EXPOSURES, and disable AE
        if isinstance(exposure, str):
            exposure = INIT_EXPOSURES[exposure][self.number]
            auto_exposure = 0

        # brightness is Black Level in DN for the 12-bit range of the detector.
        # black_level = black_level*2**4 # convert from 8-bit to 12-bit
        self.set_property('Brightness', 'Value', black_level, 'Range')
        self.set_property('Contrast', 'Value', 0, 'Range')
        self.set_property('Sharpness', 'Value', 0, 'Range')
        self.set_property('Gamma', 'Value', 100, 'Range')
        self.set_property('Gain', 'Value', gain, 'AbsoluteValue')
        self.set_property('Gain', 'Auto', 0, 'Switch')
        self.set_property('Exposure', 'Value', exposure, 'AbsoluteValue')
        self.set_property('Exposure', 'Auto', auto_exposure, 'Switch')
        self.set_property('Exposure', 'Auto Reference', 80, 'Range')
        self.set_property('Exposure', 'Auto Max Value', 10.0, 'AbsoluteValue')
        self.set_property('Exposure', 'Auto Max Auto', 0, 'Switch')
        self.set_property('Trigger', 'Enable', 0, 'Switch')
        self.set_property('Denoise', 'Value', 0, 'Range')
        self.set_property('Flip Horizontal', 'Enable', 0, 'Switch')
        self.set_property('Flip Vertical', 'Enable', 0, 'Switch')
        self.set_property('Highlight Reduction', 'Enable', 0, 'Switch')
        self.set_property('Tone Mapping', 'Enable', 0, 'Switch')
        self.set_property('Strobe', 'Enable', 0, 'Switch')
        self.set_property('Auto Functions ROI', 'Enabled', 0, 'Switch')

    def get_current_state(self):
            """Get the current property values of the camera.
            """
            # Get image info
            print('***********************************')
            print(f'Current State of Camera {self.number} ({self.name})')
            print('***********************************')
            width, height, buffer_size, bpp = self.get_image_info()
            print(f'Image size: {width} x {height} pixels')
            print(f'Image buffer size: {buffer_size} bytes')
            print(f'Bits per pixel: {bpp}')

            # Get the frame rate and color mode
            fmt = self.ic.IC_GetFormat(self.grabber)
            try:
                print(f'Color Format: {tis.SinkFormats(fmt)}')
            except:
                print(f'Color Format: {fmt}')
            fr = self.ic.IC_GetFrameRate(self.grabber)
            print(f'Frame Rate: {fr}')

            # # get number of available formats
            # vid_format_count = self.ic.IC_GetVideoFormatCount(self.grabber)
            # print(f'Number of available formats: {vid_format_count}')

            # for i in range(vid_format_count):
            #     szFormatName = (ctypes.c_char*40)()
            #     iSize = ctypes.c_int(39)
            #     iIndex = ctypes.c_int(i)
            #     ret = self.ic.IC_ListVideoFormatbyIndex(self.grabber, szFormatName, iSize, iIndex)
            #     print(f'Video format by index {iIndex.value}: {szFormatName.value}')

            # # get format
            # vid_format_idx = ctypes.c_int(17)
            # ret = self.ic.IC_GetVideoFormat(self.grabber, vid_format_idx)
            # print(f'Video format: {ret} {vid_format_idx.value}')

            self.get_property('Brightness', 'Value', 'Value', True)
            self.get_property('Contrast', 'Value', 'Value', True)
            self.get_property('Sharpness', 'Value', 'Value', True)
            self.get_property('Gamma', 'Value', 'Value', True)
            self.get_property('Gain', 'Value', 'AbsoluteValue', True)
            self.get_property('Gain', 'Auto', 'Switch', True)
            self.get_property('Exposure', 'Value', 'AbsoluteValue', True)
            self.get_property('Exposure', 'Auto', 'Switch', True)
            self.get_property('Exposure', 'Auto Reference', 'Value', True)
            self.get_property('Exposure', 'Auto Max Value', 'AbsoluteValue', True)
            self.get_property('Exposure', 'Auto Max Auto', 'Switch', True)
            self.get_property('Trigger', 'Enable', 'Switch', True)
            self.get_property('Denoise', 'Value', 'Value', True)
            self.get_property('Flip Horizontal', 'Enable', 'Switch', True)
            self.get_property('Flip Vertical', 'Enable', 'Switch', True)
            self.get_property('Highlight Reduction', 'Enable', 'Switch', True)
            self.get_property('Tone Mapping', 'Enable', 'Switch', True)
            self.get_property('Strobe', 'Enable', 'Switch', True)
            self.get_property('Auto Functions ROI', 'Enabled', 'Switch', True)

    def get_image_info(self) -> Tuple:
        """Get image info required for image capture

        :return: Image dimensions, buffer size, and bytes per pixel
        :rtype: Tuple
        """
        width = ctypes.c_long()
        height = ctypes.c_long()
        bits = ctypes.c_long()
        col_fmt = ctypes.c_int()

        self.ic.IC_GetImageDescription(self.grabber, width, height,
                                bits, col_fmt)

        if bits.value == 8:
            self.max_dn = 2**bits.value - 1
        elif bits.value == 16:
            self.max_dn = 2**12 - 2
        bpp = int(bits.value / 8.0)
        buffer_size = width.value * height.value * bits.value

        if width.value == height.value == buffer_size == bpp == 0:
            print('Warning - information 0 - open and close a video stream to initialise camera (Channel.init_camera_stream())')

        # ensure these are all updated
        self.width = width.value
        self.height = height.value
        self.buffer_size = buffer_size
        self.bpp = bpp

        return width.value, height.value, buffer_size, bpp

    def set_property(self, property: str, element: str, value, interface: str, wait: float=0.0):
        """Update the camera on-board property.

        :param property: property to update
        :type property: str
        :param element: property element to update
        :type element: str
        :param value: value to set property to
        :type value: any
        :param interface: Interface to use for setting property
        :type interface: str
        :raises ValueError: No video capture device opened
        :raises ValueError: Property is not available
        :raises ValueError: Property item element is not available
        :raises ValueError: Property element has no interface
        """
        if interface == 'Range':
            set_property_func = self.ic.IC_SetPropertyValue
            value = ctypes.c_int(value)
        elif interface == 'AbsoluteValue':
            set_property_func = self.ic.IC_SetPropertyAbsoluteValue
            value = ctypes.c_float(value)
        elif interface == 'AbsoluteValueRange':
            set_property_func = self.ic.IC_SetPropertyAbsoluteValueRange
            value = ctypes.c_float(value)
        elif interface == 'Switch':
            set_property_func = self.ic.IC_SetPropertySwitch
            value = ctypes.c_int(value)
        elif interface == 'MapStrings':
            set_property_func = self.ic.IC_SetPropertyMapStrings
            value = ctypes.c_char(value)
        elif interface == 'Button':
            set_property_func = self.ic.IC_SetPropertyOnePush
            value = ctypes.c_int(value)

        ret = set_property_func(
                self.grabber,
                property.encode("utf-8"),
                element.encode("utf-8"),
                value)

        # wait for some time for the property to update
        time.sleep(wait)

        if ret == 1:
            print(f'{property} {element} set to {value.value}')
            pass
        elif ret == -2:
            raise ValueError('No video capture device opened')
        elif ret == -4:
            raise ValueError(f'{property} is not available')
        elif ret == -5:
            raise ValueError(f'{property} item {element} is not available')
        elif ret == -6:
            raise ValueError(f'{property} {element} has no interface')
        else:
            raise ValueError(f'{property} {element} unidentified error')
        
    def set_frame_rate(self, rate: float) -> int:
        # print(f'Setting Frame Rate to : {rate} FPS')
        ret = self.ic.IC_SetFrameRate(self.grabber, ctypes.c_float(rate)) # set frame rate to 30 FPS
        if ret != 1:
            print(f'Frame Rate Error Code: {ret}')
        set_rate = self.ic.IC_GetFrameRate(self.grabber)
        print(f'Frame Rate set to : {set_rate} FPS')
        return ret

    def set_exposure(self, exposure: float) -> int:

        self.set_property('Exposure', 'Auto', 0, 'Switch')
        if exposure > 0.45:
            self.set_frame_rate(1.0)
        elif exposure <= 0.45:
            self.set_frame_rate(30.0)
        
        if exposure < 0.02:
            print('Warning: Exposures less than 0.02 s may scale unpredictably with exposure time')

        self.set_property('Exposure', 'Value', exposure, 'AbsoluteValue')

    def get_property(self, property: str, element: str, interface: str, print_state: bool=False):
        """Get the current value of a camera property."""

        if interface == 'Value':
            get_property_func = self.ic.IC_GetPropertyValue
            container = ctypes.c_long()
        elif interface == 'AbsoluteValue':
            get_property_func = self.ic.IC_GetPropertyAbsoluteValue
            container = ctypes.c_float()
        elif interface == 'AbsoluteValueRange':
            get_property_func = self.ic.IC_GetPropertyAbsoluteValueRange
            container = ctypes.c_float()
        elif interface == 'Switch':
            get_property_func = self.ic.IC_GetPropertySwitch
            container = ctypes.c_int()
        elif interface == 'MapStrings':
            get_property_func = self.ic.IC_GetPropertyMapStrings
            container = ctypes.c_char()
        elif interface == 'Button':
            get_property_func = self.ic.IC_GetPropertyOnePush
            container = ctypes.c_int()

        ret = get_property_func(
                self.grabber,
                tis.T(property),
                tis.T(element), container)

        if ret == 1:
            print(f'{property} {element}: {container.value}')
            return container.value
        elif ret == -2:
            raise ValueError('No video capture device opened')
        elif ret == -4:
            raise ValueError(f'{property} item is not available')
        elif ret == -5:
            raise ValueError(f'{property} item {element} is not available')
        elif ret == -6:
            raise ValueError(f'{property} item {element} ({interface}) has no interface')
        return container.value

    def get_exposure_range(self) -> Tuple:
        """Get the allowed range of exposures

        :return: Minimum and maximum allowed exposure (s)
        :rtype: Tuple
        """
        expmin = ctypes.c_float()
        expmax = ctypes.c_float()
        self.ic.IC_GetPropertyAbsoluteValueRange(self.grabber, tis.T("Exposure"), tis.T("Value"),
                                        expmin, expmax)
        print("Exposure range is {0} - {1}".format(expmin.value, expmax.value))

        return expmin.value, expmax.value

    def get_exposure_value(self) -> float:
        """Get the exposure time used.

        :return: Exposure (s)
        :rtype: float
        """
        container = ctypes.c_float()
        self.ic.IC_GetPropertyAbsoluteValue(self.grabber,
                                            tis.T("Exposure"),
                                            tis.T("Value"),
                                            container)
        t_exp = container.value
        return t_exp

    def find_exposure(self, init_t_exp=1.0/500, target=0.80, n_hot=10,
                      tol=1, limit=8, roi=True) -> float:
        """Find the optimal exposure time for a given peak target value.

        :param init_t_exp: initial exposure, defaults to 1.0/100
        :type init_t_exp: float, optional
        :param target: target peak image level, defaults to 150
        :type target: int, optional
        :param n_hot: if >=1: number of hot pixels allowed exceed target, 
                    if < 1: percentage of hot pixels allowed to exceed target,
                    defaults to 10
        :type n_hot: Union[int, float]
        :param tol: tolerance of meeting target criteria, defaults to 1
        :type tol: int, optional
        :param limit: number of trials to perform, defaults to 5
        :type limit: int, optional
        :param roi: use region of interest, defaults to True
        :type roi: bool, optional
        :return: exposure (seconds)
        :rtype: float
        """
        # initialise while loop
        searching = True
        trial_n = 0

        # ensure exposure setting is manual
        print('Initiating search:')
        # self.set_property('Exposure', 'Auto', 0, 'Switch')
        # self.set_property('Exposure', 'Value', init_t_exp, 'AbsoluteValue')
        self.set_exposure(init_t_exp)
        t_min, t_max = 1.0/16666, 40.0 # self.get_exposure_range()

        target = target * self.max_dn

        if n_hot < 1:
            if roi:
                n_hot = n_hot * self.camera_props['roiw'] * self.camera_props['roih']
            else:
                n_hot = n_hot * self.width * self.height

        while searching == True:
            print(f'Trial {trial_n}:')
            img_arr = self.image_capture(roi) # capture the image
            try:
                k = 1 - n_hot/(img_arr.size)
                k_quantile = np.round(np.quantile(img_arr, k)) # evaluate the quantile
                distance = np.abs(target - k_quantile)
                success = distance <= tol # check against target
                print(f'Quantile: {k_quantile}, Target: {target}')
            except: # if there is an image read error, default to a change in exposure of 10%
                k_quantile = 1.1 * t_exp_scale * set_t_exp

            if success == True:
                print(f'Success after {trial_n} trials')
                t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
                searching = False # update searcing or continue
                return t_exp

            # if the quantile is too high, increase the exposure scaling factor
            if k_quantile >= self.max_dn:
                k_quantile = 5*k_quantile
            # if the quantile is too low, decrease the exposure scaling factor
            elif k_quantile <= 0.1 * self.max_dn:
                k_quantile = k_quantile/5

            if k_quantile == 1.0/5:
                t_exp_scale = 1.5
            else:
                t_exp_scale = float(target) / float(k_quantile) # get the scaling factor
            last_t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
            new_t_exp = t_exp_scale * last_t_exp# scale the exposure
            print(f'Expected new quantile: {t_exp_scale} x {k_quantile} = {t_exp_scale*k_quantile}')
            # check the exposure is in range
            if new_t_exp < t_min:
                print(f'Exposure out of range. Setting to {t_min}')
                new_t_exp = t_min
            elif new_t_exp > t_max:
                print(f'Exposure out of range. Setting to {t_max}')
                print('WARNING: Large exposure required. Consider aborting and checking ROI and Illumination.')
                new_t_exp = t_max
            # self.set_property('Exposure', 'Value', new_t_exp, 'AbsoluteValue')
            self.set_exposure(new_t_exp)
            # check that the exposure has been set
            set_t_exp = self.get_exposure_value()
            print(f'Exposure set to {set_t_exp:0.6f}')
            trial_n+=1 # increment the counter
            failure = trial_n > limit

            if failure == True:
                print(f'Fail: {distance} DN > {tol} Tol. DN. Exiting routine.')
                t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
                searching = False
                return t_exp

    def tune_roi(self) -> None:
        """Tune the Region of Interest by fitting a 2D Gaussian to the image.
        """
        #self.find_exposure(roi=False, target=0.95, n_hot=5000, tol=50, init_t_exp=1.0/500) # aim for an overexposed image, i.e. as if thresholding has been applied.
        # self.set_exposure(20E-5)
        img = self.image_capture(roi=False)   

        crop_img = img[self.camera_props['roiy']-self.camera_props['roih']//2:self.camera_props['roiy']+2*self.camera_props['roih'], self.camera_props['roix']-self.camera_props['roiw']//2:self.camera_props['roix']+2*self.camera_props['roiw']]

        yi, xi = np.mgrid[:crop_img.shape[0], :crop_img.shape[1]]
        xyi = np.vstack([xi.ravel(), yi.ravel()])
        guess = [np.nanmax(crop_img), crop_img.shape[1]//2, crop_img.shape[0]//2, 0.001, 0.001, 0.001]
        pred_params, uncert_cov = opt.curve_fit(self.gauss2d, xyi, crop_img.ravel(), p0=guess)  

        # update the roi to the new centre
        x0, y0 = pred_params[1], pred_params[2]
        x0 = int(np.round(x0))
        y0 = int(np.round(y0))
        self.camera_props['roiy'] = y0 + self.camera_props['roiy']-self.camera_props['roih']
        self.camera_props['roix'] = x0 + self.camera_props['roix']-self.camera_props['roiw'] 

        return [self.camera_props['roiy'], self.camera_props['roix'], self.camera_props['roih'], self.camera_props['roiw']]

    @staticmethod
    def gauss2d(xy, amp, x0, y0, a, b, c):
        x, y = xy
        inner = a * (x - x0)**2 
        inner += 2 * b * (x - x0)**2 * (y - y0)**2
        inner += c * (y - y0)**2
        return amp * np.exp(-inner)

    def find_roi(self, roi_size: int=128) -> None:

        self.find_exposure(roi=False, target=0.95, n_hot=5000, tol=50, init_t_exp=1.0/500) # aim for an overexposed image, i.e. as if thresholding has been applied.
        # self.set_exposure(20E-5)
        img = self.image_capture()

        # get centre of illumination disk
        blurred = gaussian_filter(img, sigma=10)
        cntr = np.unravel_index(np.argmax(blurred, axis=None), blurred.shape)

        # opencv circle detection
        # convert image to uint8 for cv
        if self.max_dn == 2**8 - 1:
            blurred = blurred.astype(np.uint8)
        elif self.max_dn == 2**12 - 2:
            blurred = (blurred / 2**4).astype(np.uint8)
        # detect circle using hough transform
        detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 20, param2 = 30, minRadius = 30, maxRadius = 100)
        detected_circles = np.uint16(np.around(detected_circles)) # quantise
        a = detected_circles[0, 0, 0]
        b = detected_circles[0, 0, 1]
        r = detected_circles[0, 0, 2]

        # Draw the circumference of the circle.
        cv2.circle(blurred, (a, b), r, (255, 0, 0), 2)

        cntr = (b, a)
        # roi_size = r

        # set the ROI coordinates
        xlim = cntr[0]-int(roi_size/2)
        ylim = cntr[1]-int(roi_size/2)
        if xlim < 0:
            xlim = 0
        if ylim < 0:
            ylim = 0
        print(f'x: {xlim}')
        print(f'y: {ylim}')
        cv2.rectangle(blurred, (ylim, xlim), (ylim+roi_size, xlim+roi_size), (255, 255, 255), 2)

        # update the camera properties with the coodinates
        self.camera_props['roix'] = xlim
        self.camera_props['roiy'] = ylim
        self.camera_props['roiw'] = roi_size
        self.camera_props['roih'] = roi_size

        # check the ROIs
        cam_num = self.camera_props['number']
        title = f'Band {cam_num} ({self.name}) ROI Check: Context'
        self.show_image(blurred, title)
        # TODO draw on ROI on image
        img = self.image_capture(roi=True)
        title = f'Band {cam_num} ({self.name}) ROI Check: ROI'
        self.show_image(img, title)

    def set_roi_manually(self,
                        image: np.ndarray=None, 
                        roi_size: Union[int, Tuple[int,int]]=None,
                        roi_params: Tuple[int,int,int,int]=None,
                        cross_hair_is_centre: bool=False) -> None:
        """Set the ROI of the given channel manually.
        """  

        # channel_view = self.image_capture(roi=False)

        # # convert to 8-bit for cv2 view
        # if self.max_dn == 2**8 - 1:
        #     channel_view = channel_view.astype(np.uint8)
        # elif self.max_dn == 2**12 - 2:
        #     channel_view = (channel_view / 2**4).astype(np.uint8)

        # roi = cv2.selectROI(f'ROI Selection: {self.camera_props["number"]}_{int(self.camera_props["cwl"])}', channel_view, showCrosshair=True)

        # print(roi)
        # xlim = roi[1]
        # ylim = roi[0]
        # if roi_size is None:
        #     roi_size = roi[2]
        # else:
        #     print(f'Overwriting ROI size with specified value: {roi_size}')

        # # update the camera properties with the coodinates
        # self.camera_props['roix'] = xlim
        # self.camera_props['roiy'] = ylim
        # self.camera_props['roiw'] = roi_size
        # self.camera_props['roih'] = roi_size
# ------
        
        if image is None:
            img = self.image_capture(roi=False)
        else:
            img = image

        # convert to 8-bit for cv2 view
        if self.max_dn == 2**8 - 1:
            img = img.astype(np.uint8)
        elif self.max_dn == 2**12 - 2:
            img = (img / 2**4).astype(np.uint8)

        if roi_params is None:
            title = f'ROI Selection: {self.camera_props["number"]}_{int(self.camera_props["cwl"])}'            
            roi = cv2.selectROI(title, img, showCrosshair=True) # roi output is (x,y,w,h)
            # switch order of roi to (y, x, h, w)
            roi = (roi[1], roi[0], roi[3], roi[2])           
            cv2.destroyWindow(title)   

            if roi == (0,0,0,0):
                print('ROI not set - trying again')
                roi = cv2.selectROI(title, img, showCrosshair=True) # roi output is (x,y,w,h)
                # switch order of roi to (y, x, h, w)
                roi = (roi[1], roi[0], roi[3], roi[2])           
                cv2.destroyWindow(title)  
            
            if roi == (0,0,0,0):
                print('ROI not set - resorting to previous ROI')
                return [self.camera_props['roiy'], self.camera_props['roix'], self.camera_props['roih'], self.camera_props['roiw']]
            
        else:
            roi = roi_params

        if roi_size is not None:
            if isinstance(roi_size, int) :
                self.camera_props['roih'] = roi_size
                self.camera_props['roiw'] = roi_size
            else:
                self.camera_props['roih'] = roi_size[0]
                self.camera_props['roiw'] = roi_size[1]
            # update the ROI
            if cross_hair_is_centre:
                self.camera_props['roiy'] = int(roi[0]) - self.camera_props['roih']//2
                self.camera_props['roix'] = int(roi[1]) - self.camera_props['roiw']//2      
            else:
                self.camera_props['roiy'] = int(roi[0])        
                self.camera_props['roix'] = int(roi[1])
        else:
            self.camera_props['roiy'] = int(roi[0])
            self.camera_props['roix'] = int(roi[1])
            self.camera_props['roih'] = int(roi[2])
            self.camera_props['roiw'] = int(roi[3])
            if cross_hair_is_centre:
                print('Using manual ROI: cross_hair_is_centre = True ignored')

        print(f'{self.camera_props["number"]}_{int(self.camera_props["cwl"])} ROI set to: top-left corner:(y: {self.camera_props["roiy"]}, x: {self.camera_props["roix"]}), h: {self.camera_props["roih"]} w: {self.camera_props["roiw"]}')

        return [self.camera_props['roiy'], self.camera_props['roix'], self.camera_props['roih'], self.camera_props['roiw']]

    def check_roi_uniformity(self, n: int=25, ax: object=None, histo_ax: object=None) -> float:
        # check the uniformity of the ROI
        # self.find_exposure(roi=True)
        img, _, _ = self.image_capture_repeat(n=n, roi=True)
        mean = np.mean(img)
        std = np.std(img)
        uniformity = 100.0 * std / mean
        self.show_image(img, f'{self.camera_props["number"]}_{self.camera_props["cwl"]} Uniformity: {uniformity:.2f} %', ax=ax, histo_ax=histo_ax)
        print(f'ROI Uniformity: {uniformity} %') 
        return uniformity
  
    def image_capture(self, roi=False) -> np.ndarray:
        """Capture a single image from the camera.

        :param roi: Region of Interest mode, defaults to False
        :type roi: bool, optional
        :return: image data
        :rtype: np.ndarray
        """
        # self.get_current_state()
        exposure = self.get_exposure_value() # ensure that recorded exposure is correct
        framerate = self.ic.IC_GetFrameRate(self.grabber)
        print(f'Imaging with Exposure: {exposure:0.6f} s')
        print(f'Imaging with Frame Rate: {framerate} FPS')
        if exposure < 0.02:
            print('WARNING: Exposure less than 0.02 s may scale unpredictably with exposure time')
        self.ic.IC_StartLive(self.grabber,0)
        wait_time = int(np.max([5.0, 2*exposure])*1E3) # time in ms to wait to receive frame
        if self.ic.IC_SnapImage(self.grabber, wait_time) == tis.IC_SUCCESS:
            # Get the image data
            imagePtr = self.ic.IC_GetImagePtr(self.grabber)

            imagedata = ctypes.cast(imagePtr,
                                    ctypes.POINTER(ctypes.c_ubyte *
                                                self.buffer_size))

            # Create the numpy array
            image = np.ndarray(buffer=imagedata.contents,
                            dtype=np.uint8,
                            shape=(self.height, self.width, self.bpp))

            # convert to 16-bit

            if self.bpp == 2:
                image = image.astype(np.uint16)
                image = (image[:,:,0] | image[:,:,1]<<8)
                image = (image / 16).astype(np.uint16) # convert to 12-bit
            elif self.bpp == 1:
                image = image[:,:,0]
            elif self.bpp == 3:
                image = image[:,:,0]

            if roi:
                x = self.camera_props['roix']
                y = self.camera_props['roiy']
                w = self.camera_props['roiw']
                h = self.camera_props['roih']
                image = image[y:y+h, x:x+w] # NOTE - update of x and y coordinates! 21/01/2024 R Stabbins
            # print(f'+Good exposure {exposure} Image recieved')
        else:
            print(f'-Bad exposure {exposure} No image recieved in {wait_time} ms')
            image = np.full([self.height, self.width], 1, dtype=np.uint16)
            if roi:
                x = self.camera_props['roix']
                y = self.camera_props['roiy']
                w = self.camera_props['roiw']
                h = self.camera_props['roih']
                image = image[x:x+w,y:y+h]
        self.ic.IC_StopLive(self.grabber)
        return image

    def show_image(self, img_arr, title: str='', ax: object=None, histo_ax: object=None, window: str='roi_centred', draw_roi: bool=True):
        # if the image is 16 bit, convert to 8 bit for display
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.8, 4.1))
        
        if title == '':
            ax.set_title(f"{self.camera_props['number']}_{int(self.camera_props['cwl'])} Exposure: {self.get_exposure_value():.4f} s")
        else:
            ax.set_title(title)

        # default to show image as ROI with 2x size window for context
        if img_arr.shape[0] == self.camera_props['roih'] and img_arr.shape[1] == self.camera_props['roiw']:
            win_y = 0
            win_x = 0
            win_h = self.camera_props['roih']
            win_w = self.camera_props['roiw']
            draw_roi = False
        elif window == "roi":
            win_y = self.camera_props['roiy']
            win_x = self.camera_props['roix']
            win_h = self.camera_props['roih']
            win_w = self.camera_props['roiw']
        elif window == "roi_centred":
            win_y = self.camera_props['roiy'] - self.camera_props['roih']//2
            win_y = np.clip(win_y, 0, self.height-self.camera_props['roih'])
            win_x = self.camera_props['roix'] - self.camera_props['roiw']//2
            win_x = np.clip(win_x, 0, self.width-self.camera_props['roiw'])
            win_h = self.camera_props['roih']*2
            win_w = self.camera_props['roiw']*2
        elif window == "full":
            win_y = 0
            win_x = 0
            win_h = self.height
            win_w = self.width
        else:
            win_y = 0
            win_x = 0
            win_h = self.height
            win_w = self.width

        win_img = img_arr[win_y:win_y+win_h, win_x:win_x+win_w]
        extent = [win_x, win_x+win_w, win_y+win_h, win_y] # coordinates of (left, right, bottom, top)       

        disp = ax.imshow(win_img, origin='upper', extent=extent)

        im_ratio = win_img.shape[0] / win_img.shape[1]
        cbar = plt.colorbar(disp, ax=ax, fraction=0.047*im_ratio, label='DN')
        # plt.show()
        # draw window/ROI
        if draw_roi:
            rect = patches.Rectangle((self.camera_props['roix'], self.camera_props['roiy']), self.camera_props['roiw'], self.camera_props['roih'], linewidth=1, edgecolor='r', facecolor='none')        
            ax.add_patch(rect) 
            roi_img = img_arr[self.camera_props['roiy']:self.camera_props['roiy']+self.camera_props['roih'], self.camera_props['roix']:self.camera_props['roix']+self.camera_props['roiw']]
        else:
            roi_img = img_arr

        # add histogram
        if histo_ax is not None:
            counts, bins = np.histogram(roi_img[np.nonzero(np.isfinite(roi_img))], bins=128)
            histo_ax.hist(bins[:-1], bins, weights=counts,
                        label=f"{self.camera_props['number']}_{int(self.camera_props['cwl'])}",
                        log=True, fill=False, stacked=True, histtype='step')
            histo_ax.set_xlabel('Region of Interest DN')
            # plt.tight_layout()
            histo_ax.legend()

        # plt.tight_layout()

    def image_capture_repeat(self, n: int=25, 
                             roi: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """Capture n repeat images, and return the mean and standard deviation
        image, optiuonally over the ROI only.

        :param n: number of repeat images, defaults to 25
        :type n: int, optional
        :param roi: if true, return image over ROI, defaults to True
        :type roi: bool, optional
        :return: mean and standard deviation images
        :rtype: Tuple[np.ndarray, np.ndarray]
        """    
        imgs = []
        imgs = [self.image_capture(roi=roi) for i in range(n)]
        img_stk = np.dstack(imgs).astype(np.float32)
        mean = np.mean(img_stk, axis=2)
        std = np.std(img_stk, axis=2)

        if not roi:
            x = self.camera_props['roix']
            y = self.camera_props['roiy']
            w = self.camera_props['roiw']
            h = self.camera_props['roih']
            roi_img = mean[x:x+w,y:y+h]
        else:
            roi_img = mean

        return mean, std, roi_img

    def save_image(self, name, img_type, img_arr):

        exposure = self.get_property('Exposure', 'Value', 'AbsoluteValue') # note that this already ensures exposure is correct
        metadata={
            'camera': self.camera_props['number'],
            'serial': self.camera_props['serial'],
            'cwl': self.camera_props['cwl'],
            'fwhm': self.camera_props['fwhm'],
            'f-number': self.camera_props['fnumber'],
            'f-length': self.camera_props['flength'],
            'exposure': f'{exposure:.16f}', # check that string conversion is sufficient precision
            'image-type': img_type, # image or dark frame or averaged stack
            'session': self.session,
            'scene': self.scene,
            'roix': self.camera_props['roix'],
            'roiy': self.camera_props['roiy'],
            'roiw': self.camera_props['roiw'],
            'roih': self.camera_props['roih']
        }
        cam_num_str = str(self.camera_props['number'])
        cwl_str = str(int(self.camera_props['cwl']))
        channel = str(self.camera_props['number'])+'_'+cwl_str
        subject_dir = Path('..', '..', 'data', 'sessions', self.session, self.scene, channel)
        subject_dir.mkdir(parents=True, exist_ok=True)
        filename = cam_num_str+'_'+cwl_str+'_'+name+'_'+img_type
        img_file =str(Path(subject_dir, filename).with_suffix('.tif'))
        # write camera properties to TIF using ImageJ metadata
        tiff.imwrite(img_file, img_arr, imagej=True, metadata=metadata)
        print(f'Image {name} written to {img_file}')

# Camera Connection and Configuration
def start_ic() -> object:
    """Access the tisgrabber library and load the DLL

    :return: image capture object
    :rtype: object
    """
    # get the location of the tisgrabber_x64.dll file
    tis_dir = Path(tis.__file__).resolve().parents[0]
    tis_dll = str(Path(tis_dir, 'tisgrabber_x64.dll'))

    # Load/Initiate the tisgrabber library
    ic = ctypes.cdll.LoadLibrary(tis_dll)
    tis.declareFunctions(ic)
    ic.IC_InitLibrary(0)

    return ic

def load_camera_config(session_path: str=None,
                       fnumber: float=None,
                       cwl: float=None,
                       fwhm: float=None) -> Dict:
    """Load the camera configuration file

    :return: dictionary of cameras and settings
    :rtype: Dict
    """
    if session_path is None:
        camera_config_path = Path('..', '..', 'data', 'calibration', 'camera_config.csv')
    else:
        camera_config_path = Path(session_path, 'camera_config.csv')
    cameras = pd.read_csv(camera_config_path, index_col=0)
    cameras = cameras.T.astype({
        'number': 'int',
        'serial': str,
        'fnumber': float,
        'flength': float,
        'cwl': float,
        'fwhm': float,
        'width': int,
        'height': int,
        'roix': int,
        'roiy': int,
        'roiw': int,
        'roih': int}).T.to_dict()
    
    if fnumber is not None:
        for camera in cameras:
            cameras[camera]['fnumber'] = fnumber
    if cwl is not None:
        for camera in cameras:
            cameras[camera]['cwl'] = cwl
    if fwhm is not None:
        for camera in cameras:
            cameras[camera]['fwhm'] = fwhm
    
    return cameras

def get_connected_cameras(ic) -> List:
    """Get list of connected camera unique names (serial numbers)

    :param ic: ImageCapture library
    :type ic: Object
    :raises ConnectionError: Fails to connect to cameras after 5 attempts
    :return: connected camera unique device names (serial numbers).
    :rtype: List
    """
    # Get a list of connected cameras
    connected_cameras = []
    devicecount = ic.IC_GetDeviceCount()
    bad_count = 1
    while devicecount == 0:
        err_string = f'Attempt {bad_count}/5: No cameras connected - check connection and try again'
        ic.IC_MsgBox( tis.T(err_string),tis.T("Connection Error"))
        devicecount = ic.IC_GetDeviceCount()
        bad_count +=1
        if bad_count >= 5:
            raise ConnectionError('No cameras connect. Abort script and try again.')

    # Get serial names of connected cameras
    for i in range(0, devicecount):
        uniquename = tis.D(ic.IC_GetUniqueNamefromList(i))
        connected_cameras.append(uniquename)

    return connected_cameras

def connect_cameras(ic, camera_config: Dict=None) -> List:
    """Connect to available cameras, and return in a list of camera objects.

    :param cameras: camera configuration dictionary
    :type cameras: DictS
    :param ic: image capture object
    :type ic: object
    :return: list of camera objects
    :rtype: List
    """
    connected_cameras = get_connected_cameras(ic)

    # Check which cameras from config file are connected
    if camera_config is not None:
        missing_cameras = set(list(camera_config.keys())) - set(connected_cameras)
        if len(missing_cameras) > 0:
            print(f'Warning - cameras not connected: {missing_cameras}')

    cameras = []
    for camera in connected_cameras:
        if camera_config is not None:
            channel = Channel(camera, camera_config[camera], ic)
        else:
            channel = Channel(camera, None, ic)
        cameras.append(channel)

    return cameras

def configure_cameras(cameras: List[Channel], **kwargs) -> None:
    """Apply default settings to all cameras.

    :param cameras: list of camera objects
    :type cameras: List
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        camera.set_defaults(**kwargs)
        camera.get_current_state()
        print('-----------------------------------')

def find_camera_rois(cameras: List[Channel], roi_size: int=128):
    """Find the ROI for each connected camera, and update the camera properties

    :param cameras: List of connected cameras, under serial number name
    :type cameras: List
    :param roi_size: Size of region of interest, defaults to 128 pixels
    :type roi_size: int, optional
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        camera.find_roi(roi_size)
        print('-----------------------------------')

    # export_camera_config(cameras)

def set_camera_rois(cameras: Union[List[Channel], Dict[Channel, np.array]], 
                    roi_size: Union[int, Tuple[int,int]]=None,
                    roi_dict: Dict[Channel, Tuple[int,int,int,int]]=None,
                    cross_hair_is_centre: bool=False) -> Dict[Channel, Tuple[int,int,int,int]]:
    """Manually set the ROI for each connected camera, and update the camera 
    properties

    :param cameras: List[Channel] of connected cameras, under serial number name,
        or Dict[Channel, np.array] linking connected cameras to last image captured.
    :type cameras: Union[List[Channel], Dict[Channel, np.array]]
    :param roi_size: Size of region of interest, defaults to 128 pixels
    :type  Union[int, Tuple[int,int]], optional
    :param roi_dict: Dictionary of ROI parameters, defaults to None
    :type roi_dict: Dict[Channel, Tuple[int,int,int,int]], optional
    :param cross_hair_is_centre: If true, the cross hair is the centre of the ROI
    :type cross_hair_is_centre: bool, optional
    """
    if isinstance(cameras, Dict):
        camera_list = list(cameras.keys())
    else:
        camera_list = cameras

    new_roi_dict = {}
    for camera in camera_list:
        cam_num = camera.number
        if isinstance(cameras, Dict):
            last_img = cameras[camera]
        else:
            last_img = None
        if roi_dict is not None:
            roi_params = roi_dict[camera]
        else:
            roi_params = None
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        new_roi = camera.set_roi_manually(image=last_img, 
                                roi_size=roi_size,
                                roi_params=roi_params,
                                cross_hair_is_centre=cross_hair_is_centre)
        print('-----------------------------------')
        new_roi_dict[camera] = new_roi
    # export_camera_config(cameras)
    return new_roi_dict

def tune_channel_rois(cameras: List[Channel]) -> Dict[Channel, Tuple[int,int,int,int]]:
    """Tune the ROI for each connected camera, and update the camera properties

    :param cameras: List of connected cameras, under serial number name
    :type cameras: List
    """
    new_roi_dict = {}
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        new_roi_params = camera.tune_roi()
        new_roi_dict[camera] = new_roi_params
        print('-----------------------------------')

    # export_camera_config(cameras)
    return new_roi_dict

def set_camera_session(cameras: List[Channel], session: str='test_session'):
    """Set the session name for each camera.

    :param cameras: List[Channel] of connected cameras, under serial number name
    :type cameras: List[Channel]
    :param session: Description of image subject, defaults to 'test'
    :type session: str, optional
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        camera.session = session
        print(f'Session set to {session}')
        print('-----------------------------------')

def set_camera_scene(cameras: List[Channel], scene: str='test_scene'):
    """Set the scene name for each camera.

    :param cameras: List[Channel] of connected cameras, under serial number name
    :type cameras: List[Channel]
    :param scene: Description of image subject, defaults to 'test'
    :type scene: str, optional
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        camera.scene = scene
        print(f'Scene set to {scene}')
        print('-----------------------------------')

def find_channel_exposures(cameras: List[Channel], init_t_exp=0.03, target=0.8, n_hot=5,
                      tol=5, limit=16, roi=True) -> Dict:
    """Find the optimal exposure time for each camera.

    :param cameras: List[Channel] of camera objects
    :type cameras: List[Channel]
    """
    exposures = {}


    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        if init_t_exp == 'CURRENT':
            init_t_exp = camera.get_exposure_value()
        exposure = camera.find_exposure(init_t_exp, target, n_hot,
                      tol, limit, roi)
        exposures[camera.name] = exposure
        print('-----------------------------------')
    return exposures

def set_channel_exposures(cameras: List[Channel], exposures: Union[float, Dict, str]) -> None:
    """Set the exposure time for each camera.

    :param cameras: List[Channel] of camera objects
    :type cameras: List[Channel]
    :param exposures: exposure time for each camera
    :type exposures: Union[double, Dict]
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        if isinstance(exposures, float):
            expo = exposures
        elif isinstance(exposures, dict):
            try:
                expo = exposures[camera.name]
            except:
                expo = exposures[camera.number]
        elif isinstance(exposures, str):
            expo = INIT_EXPOSURES[exposures][camera.number]
        camera.set_exposure(expo)
        print(f'Exposure set to {expo} s')
        print('-----------------------------------')

def get_channel_exposures(cameras: List[Channel]) -> Dict:
    """Get the exposure time for each camera.

    :param cameras: List[Channel] of camera objects
    :type cameras: List[Channel]
    :param exposures: exposure time for each camera
    :type exposures: Union[double, Dict]
    """
    exposures = {}
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        expo = camera.get_exposure_value()
        print(f'Exposure is to {expo} s')
        exposures[camera.name] = expo
        print('-----------------------------------')
    return exposures

# Image Capture and Information Export
def capture_channel_images(cameras: List[Channel], exposures: Union[float, Dict]=None, 
                           session: str='test_session', scene: str='test_scene',
                           img_type: str='img', repeats: int=1, roi=False,
                           show_img: Union[bool, str]=False, save_img: bool=False, ax: object=None) -> None:
    """Capture a sequence of images from each camera.

    :param cameras: List[Channel] of connected camera objects
    :type cameras: List[Channel]
    :param exposures: exposure time for each camera
    :type exposures: Dict
    :param target: Description of image subject, defaults to 'test'
    :type target: str, optional
    :param img_type: Light ('img') or Dark ('drk') image, defaults to 'img'
    :type img_type: str, optional
    :param repeats: Number of repeat images to capture, defaults to 1
    :type repeats: int, optional
    :param roi: Region of interest flag, defaults to False
    :type roi: bool, optional
    :param show_img: Display image flag, defaults to False
    :type show_img: bool, optional
    :param save_img: Save image flag, defaults to False
    :type save_img: bool, optional
    """
    # TODO handle grid plot in here rather than in outside scripts
    last_imgs = {}
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        if isinstance(exposures, float):            
            camera.set_exposure(exposures)
        elif isinstance(exposures, dict):
            camera.set_exposure(exposures[camera.name])
        elif exposures == 'CURRENT':
            # do nothing - exposures are set
            pass
        
        camera.session = session # set the subject string
        camera.scene = scene
        for i in range(repeats):
            img = camera.image_capture(roi=roi)
            if show_img:
                title = ''
                if ax is not None:
                    this_ax = ax[cam_num]
                    histo_ax = ax[8]
                else:
                    this_ax = None
                    histo_ax = None
                if roi:
                    show_img ='roi'
                camera.show_image(img, title, ax=this_ax, histo_ax=histo_ax, window=show_img, draw_roi=True)
            if save_img:
                camera.save_image(str(i), img_type, img)
        print('-----------------------------------')
        last_imgs[camera] = img
    record_exposures(cameras)
    export_camera_config(cameras)
    return last_imgs

def export_camera_config(cameras: List[Channel]):
    """Export the camera properties to a csv file.

    :param cameras: Connected cameras
    :type cameras: List[Channel]
    """
    cam_info = []
    for camera in cameras:
        cam_props = list(camera.camera_props.values())
        index = list(camera.camera_props.keys())
        cam_info.append(pd.Series(cam_props, index = index, name = camera.name))
        session = camera.session
        scene = camera.scene
    cam_df = pd.concat(cam_info, axis=1)
    cam_df.sort_values('number', axis=1, ascending=True, inplace=True)
    subject_dir = Path('..', '..', 'data', 'sessions', session, scene)
    subject_dir.mkdir(parents=True, exist_ok=True)
    camera_file = Path(subject_dir, 'camera_config.csv')
    cam_df.to_csv(camera_file)

def get_camera_info(cameras: List[Channel]):
    cam_info = []
    for camera in cameras:
        cam_props = list(camera.camera_props.values())
        index = list(camera.camera_props.keys())
        cam_info.append(pd.Series(cam_props, index = index, name = camera.name))
    cam_df = pd.concat(cam_info, axis=1)
    cam_df.sort_values('number', axis=1, ascending=True, inplace=True)
    return cam_df


def record_exposures(cameras, exposures=None) -> None:
    subject_dir = Path('..', '..', 'data', 'sessions', cameras[0].session, cameras[0].scene)
    subject_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(subject_dir, 'exposure_seconds.txt')
    with open(filename, 'w') as f:
        for camera in cameras:
            cwl_str = str(int(camera.camera_props['cwl']))
            channel = str(camera.camera_props['number'])+'_'+cwl_str
            # get the exposure
            if exposures is None:
                t_exp = str(camera.get_exposure_value()) # set formatting
                camera.get_exposure_value()
            else:
                t_exp = str(exposures[camera.name]) # set formatting
            f.write(f'{channel}, {t_exp} \n')

# Camera Disconnection
def disconnect_cameras(cameras: List[Channel]) -> None:
    for camera in cameras:
        camera.ic.IC_ReleaseGrabber(camera.grabber)
        print(f'Device {camera.number} ({camera.name}) disconnected')

# Setting F-Number
def set_f_numbers_by_exp(cameras) -> None:
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        target_f_number = camera.camera_props['fnumber']
        calibration_f_number = 1.4
        # get exposure for given f_number
        t_exp_cali = camera.find_exposure(roi=True, init_t_exp=1.0/16600, limit=16, target=0.8, n_hot=10, tol=10.0)
        # compute exposure for target f_number
        t_exp_target = t_exp_cali *  target_f_number**2 / calibration_f_number**2
        # compute level for target f_number
        lvl_cali = 0.8 * camera.max_dn
        lvl_target = lvl_cali * calibration_f_number**2 / target_f_number**2
        t_exp = t_exp_cali
        searching = True
        while searching:
            title = f'{cam_num} F-Number Calibration'
            factor = np.sqrt(t_exp_target/t_exp)
            msg = f'Adjust f-number by x{factor} to get f/{target_f_number}, {t_exp_target} s exposure'
            camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
            t_exp = camera.find_exposure(roi=True, init_t_exp=t_exp_target, limit=16, target=0.8, n_hot=10, tol=10.0)
            if np.isclose(t_exp, t_exp_target, atol=0.0, rtol=0.025):
                print('Success!')
                msg = f'f/{target_f_number} achieved: {t_exp} s (Target: {t_exp_target} s)'
                camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
                camera.set_exposure(t_exp_cali)
                img_arr = camera.image_capture(roi=True)
                k = 1 - 10.0/(img_arr.size)
                lvl = np.round(np.quantile(img_arr, k)) # evaluate the quantile
                msg = f'Validation: f/{target_f_number} level: {lvl} DN (Target: {lvl_target} DN)'
                camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
                searching = False
        print('-----------------------------------')

def set_f_numbers(cameras) -> None:
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        target_f_number = camera.camera_props['fnumber']
        calibration_f_number = 1.4
        # get exposure for given f_number
        n_hot=10
        target=0.8
        t_exp_cali = camera.find_exposure(roi=True, init_t_exp=1.0/16600, limit=16, target=target, n_hot=n_hot, tol=2.0)
        # compute expected exposure for target f_number
        t_exp_target = t_exp_cali *  target_f_number**2 / calibration_f_number**2
        # compute target DN for target f_number
        lvl_cali = target * camera.max_dn
        lvl_target = lvl_cali * calibration_f_number**2 / target_f_number**2
        lvl = lvl_cali
        searching = True
        while searching:
            title = f'{cam_num} F-Number Calibration'
            factor = np.sqrt(lvl/lvl_target)
            msg = f'Adjust f-number by x{factor} to get f/{target_f_number}, {lvl_target} DN exposure from {lvl} DN'
            camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
            img_arr = camera.image_capture(roi=True)
            k = 1 - n_hot/(img_arr.size)
            lvl = np.round(np.quantile(img_arr, k)) # evaluate the quantile
            if np.isclose(lvl, lvl_target, atol=0.0, rtol=0.025):
                print('Success!')
                msg = f'f/{target_f_number} achieved: {lvl} DN (Target: {lvl_target} DN)'
                camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
                new_t_exp = camera.find_exposure(roi=True, init_t_exp=1.0/16600, limit=16, target=target, n_hot=n_hot, tol=2.0)
                msg = f'Validation: f/{target_f_number} exposure: {new_t_exp} s (Target: {t_exp_target} DN)'
                camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
                searching = False
        print('-----------------------------------')

# Experimental Functions
def adjust_spectralon_prompt(ic, i, n):
    title = 'Flat-Fielding'
    msg = f'Update Spectralon Position [{i}/{n}]'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def adjust_sample_prompt(ic):
    title = 'Sample Imaging'
    msg = f'Adjust Sample Position'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def prepare_reflectance_calibration(ic):
    title = 'Imaging Calibration Target'
    msg = 'Check Calibration Target is in place'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def prepare_geometric_calibration(ic):
    title = 'Imaging Geometric Calibration Target'
    msg = 'Check Geometric Calibration Target is in place'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))
    msg = 'Check Lens Caps are removed'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def prepare_sample_imaging(ic):
    title = 'Imaging Sample'
    msg = 'Check Sample is in place'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))
    msg = 'Check Lens Caps are removed'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def prepare_dark_acquisition(ic):
    title = 'Dark Frame Acquisition'
    msg = 'Check Lens Caps are in place'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))

def load_exposures(cameras: List[Channel], session, scene) -> Dict:
    exposures = {}
    subject_dir = Path('..', '..', 'data', 'sessions', session, scene)
    filename = Path(subject_dir, 'exposure_seconds.txt')
    exposures_in = pd.read_csv(filename, index_col=0, header=None, dtype={0: str, 1: np.float64})
    for camera in cameras:
        cam_label = str(camera.number)+'_'+str(int(camera.camera_props['cwl']))
        exposures[camera.name] = exposures_in.loc[cam_label].values[0]
    return exposures

def check_channel_roi_uniformity(cameras: List[Channel], n: int=25, ax: object=None) -> None:
    """Check the uniformity of the ROI for each camera.

    :param cameras: List[Channel] of connected camera objects
    :type cameras: List[Channel]
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        if ax is not None:
            this_ax = ax[cam_num]
            histo_ax = ax[8]
        else:
            this_ax = None
            histo_ax = None
        camera.check_roi_uniformity(n=n, ax=this_ax, histo_ax=histo_ax)
        print('-----------------------------------')

def find_camera_bands(connected_cameras: List[Channel], cameras: Dict) -> Dict:
    """Find the band number for each connected camera, and update the camera
    properties dictionary.

    :param connected_cameras: List[Channel] of connected cameras, under serial number name
    :type cameras: List[Channel]
    :param cameras: Camera properties dictionary
    :type cameras: Dict
    :return: Dictionary of camera properties, with serial number attached
    :rtype: Dict
    """
    for camera in connected_cameras:
        camera.ic.IC_StartLive(camera.grabber,1)
        camera.ic.IC_MsgBox(tis.T('Find the band number by waving in front of each camera'), tis.T('Camera Configuration'))
        camera.ic.IC_StopLive(camera.grabber,1)
        band_number = input(prompt='Enter band number e.g. "3"')
        band_label = f'Band{band_number}'
        cameras[band_label]['serial'] = camera.name
        cameras[camera.name] = cameras.pop(band_label)
    return cameras

def set_focus(cameras) -> None:
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        camera.ic.IC_StartLive(camera.grabber, 1)
        title = f'{cam_num} Focus Calibration'
        msg = f'Adjust Focus for Device {cam_num}'
        camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
        camera.ic.IC_StopLive(camera.grabber)
        print('-----------------------------------')

def check_f_numbers(cameras) -> None:
    # check against first camera
    camera = cameras[0]
    cam_num = camera.number
    print('-----------------------------------')
    print(f'Device {cam_num}')
    print('-----------------------------------')
    t_exp_cali = camera.find_exposure(init_t_exp=1.0/16600, target=0.80, n_hot=10, tol=0.5, limit=10, roi=True)

    # prepare histogram
    fig, ax = plt.subplots(1,1)
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        # camera.set_property('Exposure', 'Value', t_exp_cali, 'AbsoluteValue')
        # camera.set_property('Exposure', 'Auto', 0, 'Switch')
        camera.set_exposure(t_exp_cali)
        img = camera.image_capture(roi=True)
        # plot histogram
        counts, bins = np.histogram(img[np.nonzero(np.isfinite(img))], bins=128)
        ax.hist(bins[:-1], bins, weights=counts,
                      label=f'({cam_num})',
                      log=True, fill=False, stacked=True, histtype='step')
    # show legend
    ax.legend()
    plt.show()

class CallbackUserdata(ctypes.Structure):
    """ Example for user data passed to the callback function.
    """
    def __init__(self, ):
        self.unsused = ""
        self.devicename = ""
        self.connected = False

def frameReadyCallback(hGrabber, pBuffer, framenumber, pData):
    # Maybe do something here.
    return

def deviceLostCallback(hGrabber, userdata):
    """ This a device lost callback function. Called, if the camera disconnects.
    This function runs in the Grabber thread, not in the main thread.
    :param: hGrabber: This is the real pointer to the grabber object. Do not use.
    :param: pData : Pointer to additional user data structure
    """
    userdata.connected = False
    print("Device {} lost".format(userdata.devicename))
