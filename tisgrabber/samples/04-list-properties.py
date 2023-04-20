import ctypes
import tisgrabber as tis

ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll")
tis.declareFunctions(ic)

ic.IC_InitLibrary(0)

hGrabber = ic.IC_ShowDeviceSelectionDialog(None)

if(ic.IC_IsDevValid(hGrabber)):
    ic.IC_printItemandElementNames(hGrabber)
else:
    print("No device opened.")

ic.IC_ReleaseGrabber(hGrabber)
