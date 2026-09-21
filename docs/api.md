# API reference

## Stable inference API

```{eval-rst}
.. autofunction:: geoinference.estimate
.. autofunction:: geoinference.estimate_from_file
.. autofunction:: geoinference.read_frames

.. automodule:: geoinference.designs
   :members:

.. automodule:: geoinference.types
   :members:
   :exclude-members: __init__
```

## Experimental validation API

These modules diagnose dependence and validate collection designs. Their
interfaces may change before the stable inference API does.

```{eval-rst}
.. automodule:: geoinference.spatial
   :members:

.. automodule:: geoinference.simulate
   :members:
   :exclude-members: __init__

.. automodule:: geoinference.pipeline
   :members:
   :exclude-members: __init__
```
