### SYSTEM ###
try:
    import os as system
except ImportError as e:
    system = None
    print(e)

### DATA ###
try:
    import csv as csv_engine
except ImportError as e:
    csv_engine = None
    print(e)

try:
    import json as json_engine
except ImportError as e:
    json_engine = None
    print(e)

try:
    import pickle as pickle_engine
except ImportError as e:
    pickle_engine = None
    print(e)

try:
    import copy as clone
except ImportError as e:
    clone = None
    print(e)

### MATH ###
try:
    import numpy as cpu_engine
except ImportError as e:
    cpu_engine = None
    print(e)

try:
    import cupy as gpu_engine
except ImportError as e:
    gpu_engine = None
    print(e)

### ANNOTATIONS ###
try:
    from typing import override as override_from_parent
except ImportError as e:
    override_from_parent = None
    print(e)

try:
    from dataclasses import dataclass as dto, field
except ImportError as e:
    dto = None
    print(e)

### DATETIME ###
try:
    from datetime import datetime as date_time_engine
except ImportError as e:
    datetime = None
    print(e)

### PLOTTING ###
try:
    import matplotlib.pyplot as pltotting
except ImportError as e:
    plotting = None
    print(e)

### ENUM ###
try:
    from enum import Enum, auto
except ImportError as e:
    enum, auto = None, None
    print(e)
