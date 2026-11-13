"""LanWind: radar and forecast conditioned residual ConvLSTM."""
__version__ = '1.0.0'
__all__ = ['LanWind']

def __getattr__(name):
    if name=='LanWind':
        from .model import SixHourConvLSTM
        return SixHourConvLSTM
    raise AttributeError(name)
