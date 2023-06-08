"""A library of functions for controlling the OROCHI Simulator.

Roger Stabbins
Rikkyo University
21/04/2023
"""

import ctypes
from datetime import date
from pathlib import Path
import time
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import tifffile as tiff
import tisgrabber as tis

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
            # pass
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
        print(f'Setting Frame Rate to : {rate} FPS')
        ret = self.ic.IC_SetFrameRate(self.grabber, ctypes.c_float(rate)) # set frame rate to 30 FPS
        print(f'set frame rate err: {ret}')
        set_rate = self.ic.IC_GetFrameRate(self.grabber)
        print(f'Frame Rate set to : {set_rate} FPS')
        return ret

    def set_defaults(self, exposure=1.0/100, auto_exposure=1, black_level=0):
        """Set default properties for each camera.

        :param exposure: exposure time (seconds), defaults to 1.0/100
        :type exposure: float, optional
        :param auto_exposure: AE activation flag, defaults to 1
        :type auto_exposure: int, optional
        :param black_level: black offset level, in 8-bit DN range, defaults to
            26 (10% of the range, for linearity)
        :type black_level: int, optional
        """
        # 8-bit
        # vid_format = "Y800 (1920x1200)"
        # sink_format_id = 0
        # # 12-bit
        vid_format = "Y16 (1920x1200)"
        sink_format_id = 4
        ret = self.ic.IC_SetVideoFormat(self.grabber, tis.T(vid_format))
        print(ret)
        print(f'Video Format set to : {vid_format}')
        ret = self.ic.IC_SetFormat(self.grabber, ctypes.c_int(sink_format_id))
        print(ret)
        print(f'Sink Format set to : "{tis.SinkFormats(sink_format_id)}"')
        ret = self.ic.IC_SetFrameRate(self.grabber, ctypes.c_float(30.0)) # set frame rate to 30 FPS
        print(f'Frame Rate set to : {30.0} FPS')
        self.init_camera_stream()
        self.get_image_info()
        # brightness is Black Level in DN for the 12-bit range of the detector.
        # black_level = black_level*2**4 # convert from 8-bit to 12-bit
        self.set_property('Brightness', 'Value', black_level, 'Range')
        self.set_property('Contrast', 'Value', 0, 'Range')
        self.set_property('Sharpness', 'Value', 0, 'Range')
        self.set_property('Gamma', 'Value', 100, 'Range')
        self.set_property('Gain', 'Value', 4.27, 'AbsoluteValue')
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
            print(f'{property} current {element}: {container.value}')
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

    def get_current_state(self):
        """Get the current property values of the camera.
        """
        # Get image info
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

    def image_capture(self, roi=False) -> np.ndarray:
        """Capture a single image from the camera.

        :param roi: Region of Interest mode, defaults to False
        :type roi: bool, optional
        :return: image data
        :rtype: np.ndarray
        """
        # self.get_current_state()
        self.ic.IC_StartLive(self.grabber,0)
        exposure = self.get_exposure_value() # ensure that recorded exposure is correct
        print(f'Imaging with Exposure: {exposure} s')
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
                image = image[x:x+w,y:y+h]
            print(f'+Good exposure {exposure} Image recieved')
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

    def show_image(self, img_arr, title):
        # if the image is 16 bit, convert to 8 bit for display
        fig, ax = plt.subplots(figsize=(5.8, 4.1))
        disp = ax.imshow(img_arr, origin='lower')
        ax.set_title(title)
        plt.colorbar(disp, ax=ax)
        plt.tight_layout()
        plt.show()

    def find_exposure(self, init_t_exp=1.0/500, target=0.80, n_hot=10,
                      tol=1, limit=8, roi=True) -> float:
        """Find the optimal exposure time for a given peak target value.

        :param init_t_exp: initial exposure, defaults to 1.0/100
        :type init_t_exp: float, optional
        :param target: target peak image level, defaults to 150
        :type target: int, optional
        :param n_hot: number of hot pixels allowed exceed target, defaults to 10
        :type n_hot: int, optional
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
        self.set_property('Exposure', 'Auto', 0, 'Switch')
        self.set_property('Exposure', 'Value', init_t_exp, 'AbsoluteValue')
        t_min, t_max = 1.0/16666, 50.0 # self.get_exposure_range()

        target = target * self.max_dn

        while searching == True:
            print(f'Trial {trial_n}:')
            img_arr = self.image_capture(roi) # capture the image
            try:
                k = 1 - n_hot/(img_arr.size)
                k_quantile = np.round(np.quantile(img_arr, k)) # evaluate the quantile
                distance = np.abs(target - k_quantile)
                success = distance <= tol # check against target
                print(f'Quantile: {k_quantile}, Target: {target}')
            except:
                k_quantile = 1.1 * t_exp_scale * set_t_exp

            if success == True:
                print(f'Success after {trial_n} trials')
                t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
                searching = False # update searcing or continue
                return t_exp

            if k_quantile >= self.max_dn:
                k_quantile = 5*k_quantile
            elif k_quantile <= 0.1 * self.max_dn:
                k_quantile = k_quantile/5

            if k_quantile == 1.0/5:
                t_exp_scale = 1.5
            else:
                t_exp_scale = float(target) / float(k_quantile) # get the scaling factor
            last_t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
            new_t_exp = t_exp_scale * last_t_exp# scale the exposure
            # check the exposure is in range
            if new_t_exp < t_min:
                print(f'Exposure out of range. Setting to {t_min}')
                new_t_exp = t_min
            elif new_t_exp > t_max:
                print(f'Exposure out of range. Setting to {t_max}')
                new_t_exp = t_max
            self.set_property('Exposure', 'Value', new_t_exp, 'AbsoluteValue')
            # check that the exposure has been set
            set_t_exp = self.get_exposure_value()
            print(f'Exposure set to {set_t_exp} (err of {new_t_exp - set_t_exp}')
            trial_n+=1 # increment the counter
            failure = trial_n > limit

            if failure == True:
                print(f'Failure to satisfy tolerance. Exiting routine.')
                t_exp = self.get_property('Exposure', 'Value', 'AbsoluteValue')
                searching = False
                return t_exp

    def find_roi(self, roi_size: int=128) -> None:

        self.find_exposure(roi=False, target=0.95, n_hot=5000, tol=50) # aim for an overexposed image, i.e. as if thresholding has been applied.
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

    def check_roi_uniformity(self) -> float:
        # check the uniformity of the ROI
        self.find_exposure(roi=False)
        img = (self.image_capture(roi=True)).astype(np.float64)
        for i in range(24):
            img += self.image_capture(roi=True)
        img = img / 25
        self.show_image(img, 'ROI Uniformity')
        mean = np.mean(img)
        std = np.std(img)
        print(f'ROI Uniformity: {100.0 * std / mean} %')
        return std / mean

    def simple_linearity_check(self, n_exp: int=50) -> None:
        """Run a simple program to check linearity, by imaging the
        reflectance calibration target at step exposures from minimum
        up to saturation.

        :param n_exp: number of exposure steps to use, defaults to 50
        :type n_exp: int, optional
        """
        self.set_property('Exposure', 'Auto', 0, 'Switch')
        # set the minimum exposure as initial
        t_min = 6.5E-5
        t_exp = t_min
        # set the saturation limit
        sat_limit = 200
        mean = 0

        t_exps = []
        means = []
        count = 0
        factor = 2.0
        while count < n_exp:
            # set exposure
            if t_exp < t_min:
                t_exp = t_min
            self.set_property('Exposure', 'Value', t_exp, 'AbsoluteValue', wait=1.0)
            # capture image
            img = self.image_capture(roi=True)
            # get estimate of mean over target ROI
            mean = np.mean(img)
            print(f'Mean Signal: {mean} DN')
            # record t_exp, record mean
            t_exps.append(t_exp)
            means.append(mean)
            # set next exposure
            t_exp = factor*t_exp
            # update counter
            if mean >=sat_limit:
                factor = 0.9
            count+=1
        data = pd.DataFrame(data={'t_exp': t_exps, 'mean': means})
        data.sort_values('t_exp', inplace=True)
        data.plot('t_exp', 'mean', logx=True, logy=True)
        plt.show()
        data.plot('t_exp', 'mean')
        plt.show()

    def save_image(self, name, subject, img_type, img_arr):

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
            'subject': subject,
            'roix': self.camera_props['roix'],
            'roiy': self.camera_props['roiy'],
            'roiw': self.camera_props['roiw'],
            'roih': self.camera_props['roih']
        }
        cwl_str = str(int(self.camera_props['cwl']))
        channel = str(self.camera_props['number'])+'_'+cwl_str
        subject_dir = Path('..', 'data', subject, channel)
        subject_dir.mkdir(parents=True, exist_ok=True)
        filename = cwl_str+'_'+name+'_'+img_type
        img_file =str(Path(subject_dir, filename).with_suffix('.tif'))
        # write camera properties to TIF using ImageJ metadata
        tiff.imwrite(img_file, img_arr, imagej=True, metadata=metadata)
        print(f'Image {name} written to {img_file}')

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

def load_camera_config() -> Dict:
    """Load the camera configuration file

    :return: dictionary of cameras and settings
    :rtype: Dict
    """
    cameras = pd.read_csv('camera_config.csv', index_col=0)
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

def configure_cameras(cameras: List) -> None:
    """Apply default settings to all cameras.

    :param cameras: list of camera objects
    :type cameras: List
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num} ({camera.name})')
        print('-----------------------------------')
        camera.set_defaults()
        camera.get_current_state()
        print('-----------------------------------')

def find_camera_bands(connected_cameras: List, cameras: Dict) -> Dict:
    """Find the band number for each connected camera, and update the camera
    properties dictionary.

    :param connected_cameras: List of connected cameras, under serial number name
    :type cameras: List
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

def find_camera_rois(cameras: List, roi_size: int=128):
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

    export_camera_config(cameras)

def export_camera_config(cameras: List):
    """Export the camera properties to a csv file.

    :param cameras: Connected cameras
    :type cameras: List
    """
    cam_info = []
    for camera in cameras:
        cam_props = list(camera.camera_props.values())
        index = list(camera.camera_props.keys())
        cam_info.append(pd.Series(cam_props, index = index, name = camera.name))
    cam_df = pd.concat(cam_info, axis=1)
    cam_df.sort_values('number', axis=1, ascending=True, inplace=True)
    camera_file = 'camera_config.csv'
    cam_df.to_csv(camera_file)

def prepare_reflectance_calibration(ic):
    title = 'Imaging Calibration Target'
    msg = 'Check Calibration Target is in place'
    ic.IC_MsgBox(tis.T(msg), tis.T(title))
    msg = 'Check Lens Caps are removed'
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

def find_channel_exposures(cameras: List, init_t_exp=1.0, target=150, n_hot=5,
                      tol=1, limit=10, roi=True) -> Dict:
    """Find the optimal exposure time for each camera.

    :param cameras: list of camera objects
    :type cameras: List
    """
    exposures = {}
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        exposure = camera.find_exposure(init_t_exp, target, n_hot,
                      tol, limit, roi)
        exposures[camera.name] = exposure
        print('-----------------------------------')
    return exposures

def check_channel_roi_uniformity(cameras: List) -> None:
    """Check the uniformity of the ROI for each camera.

    :param cameras: List of connected camera objects
    :type cameras: List
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        camera.check_roi_uniformity()
        print('-----------------------------------')

def check_channel_linearity(cameras: List, n_exp: int=50) -> None:
    """Check the channel linearity.

    :param cameras: List of connected camera objects
    :type cameras: List
    :param n_exp: Number of exposure steps to take, defaults to 50
    :type n_exp: int, optional
    """
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        camera.simple_linearity_check(n_exp)
        print('-----------------------------------')

def capture_channel_images(cameras: List, exposures: Dict, subject: str='test',
                           img_type: str='img', repeats: int=1, roi=False,
                           show_img: bool=False, save_img: bool=False) -> None:
    """Capture a sequence of images from each camera.

    :param cameras: List of connected camera objects
    :type cameras: List
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

    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        camera.set_property('Exposure', 'Value', exposures[camera.name], 'AbsoluteValue')
        camera.set_property('Exposure', 'Auto', 0, 'Switch')
        for i in range(repeats):
            img = camera.image_capture(roi=roi)
            if show_img:
                title = f'Device {cam_num} {subject} #{i}'
                camera.show_image(img, title)
            if save_img:
                camera.save_image(str(i), subject, img_type, img)
        print('-----------------------------------')

def record_exposures(cameras, exposures, subject) -> None:
    for camera in cameras:
        cwl_str = str(int(camera.camera_props['cwl']))
        channel = str(camera.camera_props['number'])+'_'+cwl_str
        subject_dir = Path('..', 'data', subject, channel)
        subject_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(subject_dir, 'exposure_seconds.txt')
        with open(filename, 'w') as f:
                t_exp = str(exposures[camera.name]) # set formatting
                f.write(t_exp)

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
    t_exp_cali = camera.find_exposure(tol=0.5,roi=True)

    # prepare histogram
    fig, ax = plt.subplots(1,1)
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        camera.set_property('Exposure', 'Value', t_exp_cali, 'AbsoluteValue')
        camera.set_property('Exposure', 'Auto', 0, 'Switch')
        img = camera.image_capture(roi=True)
        # plot histogram
        counts, bins = np.histogram(img[np.nonzero(np.isfinite(img))], bins=128)
        ax.hist(bins[:-1], bins, weights=counts,
                      label=f'({cam_num})',
                      log=True, fill=False, stacked=True, histtype='step')
    # show legend
    ax.legend()
    plt.show()


def set_f_numbers(cameras) -> None:
    for camera in cameras:
        cam_num = camera.number
        print('-----------------------------------')
        print(f'Device {cam_num}')
        print('-----------------------------------')
        target_f_number = camera.camera_props['fnumber']
        calibration_f_number = 1.4
        # get exposure for given f_number
        t_exp_cali = camera.find_exposure(tol=0.5,roi=True)
        # compute exposure for target f_number
        t_exp_target = t_exp_cali *  target_f_number**2 / calibration_f_number**2
        t_exp = t_exp_cali
        searching = True
        while searching:
            title = f'{cam_num} F-Number Calibration'
            factor = np.sqrt(t_exp_target/t_exp)
            msg = f'Adjust f-number by x{factor} to get f/{target_f_number}, {t_exp_target} s exposure'
            camera.ic.IC_MsgBox(tis.T(msg), tis.T(title))
            t_exp = camera.find_exposure(tol=0.5,roi=True)
            if np.isclose(t_exp, t_exp_target, atol=0.0, rtol=0.02):
                print('Success!')
                searching = False
        print('-----------------------------------')

def load_exposures(cameras, subject) -> Dict:
    subject_dir = Path('..', 'data', subject)
    channels = subject_dir.glob('*')
    exposures = {}
    cam_lut = {str(c.number): c.name for c in cameras}
    for channel in channels:
        filename = Path(channel, 'exposure_seconds.txt')
        cam_num = channel.name.split('_')[0]
        cam_name = cam_lut[cam_num]
        with open(filename, 'r') as f:
            exposures[cam_name] = float(f.read())
    return exposures

def disconnect_cameras(cameras: List) -> None:
    for camera in cameras:
        camera.ic.IC_ReleaseGrabber(camera.grabber)
        print(f'Device {camera.number} ({camera.name}) disconnected')

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
