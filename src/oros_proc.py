"""A Library of Classes and Functions for processing images captured with
the OROCHI Laboratory Simulator, OROS

Some notes in progress:
- the system should be able to host multiple target images in one instance
- this will be achieved by each channel hosting multiple Images
- in turn, each StereoPair and each MultiChannelSystem will have to keep track
of these multiple images. So we need some kind of image list that is shared
between all of the objects, or perhaps some way of calling up common images
in the list of channel images.

Roger Stabbins
Rikkyo University
18/8/2023
"""

class Image:
    """Super class for handling images.
    """
    def __init__(self):
        pass
    def image_load(self):
        pass


class LightImage(Image):
    "Class for handling light images, inherits from Image class"
    pass

class DarkImage(Image):
    "Class for handling dark images, inherits from Image class"
    pass

class Channel:
    """Class for handling a channel of a multi-channel system, defined by
    spectral properties and pose."""
    def __init__(self):
        pass

class StereoPair:
    """Class for handling stereo pairs of a multi-channel system, composed of
    2 channels.
    """
    def __init__(self, channel_a, channel_b):
        self.ch_a = channel_a
        self.ch_b = channel_b

class MultiChannelSystem:
    """Class for handling a multi-channel system, e.g. OROS.
    (in principal, could handle PanCam.)
    """
    def __init__(self, channels):
        self.channels = channels # a list of channel objects
        self.stereo_pairs = []
        # build a list of stereo pairs from the channels
