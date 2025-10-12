# Simulated LED strip driver

from .hw_base import HardwareDriver

class LEDStrip(HardwareDriver):
    # Set device name
    name = 'led_strip'

    def __init__(self, simulate:bool=True, port:str=None):
        # Save fields
        self.simulate = simulate
        self.port = port
        # Internal state for color and brightness
        self.state = {'color': '#000000', 'brightness': 0}
        # Armed flag for dangerous ops
        self.armed = False

    def schema(self):
        # Describe supported actions
        return {
            'actions': [
                {'name': 'status', 'args': []},
                {'name': 'set_brightness', 'args': ['brightness']},
                {'name': 'set_color', 'args': ['hex']},
                {'name': 'arm', 'args': []},
            ]
        }

    def call(self, action:str, **kwargs):
        # Dispatch by action name
        if action=='status':
            return {'state': self.state, 'armed': self.armed, 'simulate': self.simulate}
        if action=='set_brightness':
            # Clamp brightness between 0 and 100
            b = int(kwargs.get('brightness', 0))
            b = 0 if b<0 else 100 if b>100 else b
            self.state['brightness'] = b
            return {'ok': True, 'state': self.state}
        if action=='set_color':
            # Basic color set
            hx = str(kwargs.get('hex', '#000000'))
            if not hx.startswith('#'):
                hx = '#'+hx
            self.state['color'] = hx
            return {'ok': True, 'state': self.state}
        if action=='arm':
            # Set armed flag
            self.armed = True
            return {'armed': True}
        # Unknown action
        return {'error': f'unknown led_strip action: {action}'}
