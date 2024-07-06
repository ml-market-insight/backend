import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
import certifi
import json
from fastapi import FastAPI, APIRouter, HTTPException
import ssl
from urllib.request import urlopen
from pymongo import MongoClient


global keys
keys = ["kRUw6nhgCGQPVpbL00dWh4DN0KcQUNJc", "OIX7rzkEIms1sXSLYb4gx0HohTWQXCxd", "awLSkLYhoTipHBricJaH01OZBezcs1cu", 
        "eVQHEMHlTDuRalM6Kab7pdEAqhgbAk5w", "FzNA2EPeZQmjXd0ibvSyRtybkZuIApKN"]
global key_index
key_index = 0
