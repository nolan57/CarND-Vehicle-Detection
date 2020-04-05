from helper import *
import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

svc_pkl = pickle.load(open('svc_pkl_YCrCb.pkl','rb'))

svc = svc_pkl['svc']
X_scaler = svc_pkl['X_scaler']
colorspace = svc_pkl['colorspace']
orient = svc_pkl['orient']
pix_per_cell = svc_pkl['pix_per_cell']
cell_per_block = svc_pkl['cell_per_block']
spatial_size = svc_pkl['spatial_size']
hist_bins = svc_pkl['hist_bins']
hist_range = svc_pkl['hist_range']

accs = []

#scale = 1.5
#orient = 9
#pix_per_cell = 8
#cell_per_block = 2
#spatial_size = (16, 16)
#hist_bins = 16

def vd_video(image):
    
    global svc, X_scaler, colorspace
    global orient, pix_per_cell, cell_per_block
    global spatial_size, hist_bins    
    global counter, boxes
    global accs

    ystart = 350
    xstart = 0
    #ystop = 650
    yend = 650

    windows_size = 8

    scale = 1.5

    overlap = 0.5

    #draw_img, bbox_list = find_cars(image,
    #                                ystart, ystop, scale, colorspace,
    #                                svc, X_scaler,
    #                                orient, pix_per_cell, cell_per_block,
    #                                spatial_size, hist_bins)

    draw_img, bbox_list = msw_find_cars(image,
                                        ystart, yend, xstart, windows_size, scale, overlap,
                                        colorspace,
                                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    addheat = add_heat(heatmap,bbox_list)
    thresh_heat = apply_threshold(addheat,6)
    clpheat = np.clip(heatmap, 0, 255)

    labels = label(clpheat)
    if len(bbox_list) != 0:
        accs.append(len(labels)/len(bbox_list))
        avg_acc = np.mean(accs)
        avg_acc_pkl_name = 'avg_acc_' + colorspace +'_final.pkl'
        pickle.dump(avg_acc, open(avg_acc_pkl_name, 'wb'))

    output = draw_labeled_bboxes(np.copy(image), thresh_heat)
    
    #return draw_img, addheat, thresh_heat, output :this return for debug
    return output

def process_on_video(input_file, output_file):

    clip = VideoFileClip(input_file).subclip(40, 43)
    vd_video_output = clip.fl_image(vd_video)
    vd_video_output.write_videofile(output_file, audio=False)

if __name__ == '__main__' :

    process_on_video('./project_video.mp4', './project_video_YCrCb_msw2.mp4')