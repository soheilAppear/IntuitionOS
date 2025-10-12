# GPU info plugin using nvidia-ml-py if available

try:
    import pynvml as _nv  # some systems still ship this name
except Exception:
    try:
        import nvidia_smi as _nv  # rarely available
    except Exception:
        _nv = None

from .hw_base import HardwareDriver

class GPUNVML(HardwareDriver):
    name = 'gpu_nvml'

    def __init__(self, enabled:bool=True):
        # Flag only
        self.enabled = enabled

    def schema(self):
        # Single status action
        return {'actions': [{'name': 'status', 'args': []}]}

    def call(self, action:str, **kwargs):
        # Only support status
        if action!='status':
            return {'error': 'only status is supported'}
        if not self.enabled or _nv is None:
            return {'enabled': False}
        # Try to read basic info
        try:
            _nv.nvmlInit()
            count = _nv.nvmlDeviceGetCount()
            gpus=[]
            for i in range(count):
                h = _nv.nvmlDeviceGetHandleByIndex(i)
                name = _nv.nvmlDeviceGetName(h).decode() if hasattr(_nv.nvmlDeviceGetName(h),'decode') else _nv.nvmlDeviceGetName(h)
                mem = _nv.nvmlDeviceGetMemoryInfo(h)
                gpus.append({'index': i, 'name': name, 'mem_total': int(mem.total), 'mem_used': int(mem.used)})
            _nv.nvmlShutdown()
            return {'gpus': gpus}
        except Exception as e:
            return {'error': str(e)}
