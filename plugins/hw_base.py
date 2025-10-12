# Base class for simple hardware like devices

class HardwareDriver:
    # Default device name
    name = 'abstract'

    def schema(self):
        # Return a small schema for UI
        return {'actions': []}

    def call(self, action:str, **kwargs):
        # Must be implemented by subclasses
        raise NotImplementedError
