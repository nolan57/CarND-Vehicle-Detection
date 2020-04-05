import pickle
from helper import *

colorspaces = []

colorspaces.append('HLS')
colorspaces.append('HSV')
colorspaces.append('LUV')
colorspaces.append('YUV')
colorspaces.append('YCrCb')
#colorspace = 'HSV'
#colorspace = 'LUV'
#colorspace = 'YUV'
#colorspace = 'YCrCb'

orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
#spatial_size = (16, 16)
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)

svc_pkl = {}
for colorspace in colorspaces:

    print('colorspace：', colorspace)

    cars, notcars = load_data()
    svc, X_scaler, acc = extract_and_train(cars, notcars, colorspace, orient, pix_per_cell,
                      cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

    svc_pkl['svc'] = svc
    svc_pkl['X_scaler'] = X_scaler
    svc_pkl['colorspace'] = colorspace
    svc_pkl['orient'] = orient
    svc_pkl['pix_per_cell'] = pix_per_cell
    svc_pkl['cell_per_block'] = cell_per_block
    svc_pkl['spatial_size'] = spatial_size
    svc_pkl['hist_bins'] = hist_bins
    svc_pkl['hist_range'] = hist_range
    svc_pkl['acc'] = acc

    svc_pkl_name = 'svc_pkl_' + colorspace +'_32.pkl'
    pickle.dump(svc_pkl, open(svc_pkl_name, 'wb'))