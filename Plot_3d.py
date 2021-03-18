#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.


# run the script by passing two npz in the commandline. The library for ploting in called plotly



import ctypes
import numpy as np
import deep_ls.data
import sys
import OpenGL.GL as gl
sys.path.append('/home/philippgfriedrich/DeepSDF/Pangolin/build/src/')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly as py



def plot_3D(xyz_data, sdf, epoch, color, camera, save_dir,flag_plot=False, flag_screenshot=True,fig=None, row=None,col=None,scene=None):
    pio.orca.config.port = 8960

    cbarlocs=[[None,None],[0.3,0.85],[0.3,0.85],[1.0,0.85],[0.3,0.5],[0.3,0.5],[1.0,0.5],[0.3,0.15],[0.3,0.15],[1.0,0.15]]
    cmin=color[0]
    cmax = color[1]
    # if(scene==5 or scene==6):
    #     cmin=-0.1
    #     cmax=0

    if fig is None:
        fig = go.Figure()
        fig = make_subplots(rows=4, cols=2, specs=[ [{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}], 
                                                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}] ],
                        subplot_titles=(
                            'All_Gt', 'All_Pred' 'Negative_Gt', 'Negativ_Pred', 'Zero GT', 'Zero Pred' 'Pos. GT', 'Pos. Pred'),
                        vertical_spacing=0.05, horizontal_spacing=0.001)
    fig.add_trace(go.Scatter3d(x=xyz_data[:, 0], y=xyz_data[:, 1], z=xyz_data[:, 2],
                               mode='markers',
                               name=f'Plot_{scene}',
                               scene=f'scene{scene}',
                               marker=dict(
                                   size=1.5,
                                   opacity=1.0,
                                   # showscale=True,
                                   cmin=cmin,
                                   cmax=cmax,
                                   colorscale="rainbow",
                                   color=sdf.squeeze(),
                                   colorbar= dict(len=0.3, x=cbarlocs[scene][0], y=cbarlocs[scene][1]),
                               )
                               ), row,col
                  )
    camera = dict(
        up=dict(x=camera[0][0], y=camera[0][1], z=camera[0][2]),
        eye=dict(x=camera[1][0], y=camera[1][1], z=camera[1][2])
    )

    fig.update_layout(title=go.layout.Title(text=f'{color[2]}: subbatch_{epoch}'),margin=dict(r=10, l=10, b=25, t=30))
    if (scene == 1):
        fig.update_layout(scene_camera=camera, scene_aspectmode='data')
    elif (scene == 2):
        fig.update_layout(scene2_camera=camera, scene2_aspectmode='data')
    elif (scene == 3):
        fig.update_layout(scene3_camera=camera, scene3_aspectmode='data')
    elif (scene == 4):
        fig.update_layout(scene4_camera=camera, scene4_aspectmode='data')
    elif (scene == 5):
        fig.update_layout(scene5_camera=camera, scene5_aspectmode='data')
    elif (scene == 6):
        fig.update_layout(scene6_camera=camera, scene6_aspectmode='data')
    elif (scene == 7):
        fig.update_layout(scene7_camera=camera, scene7_aspectmode='data')
    elif (scene == 8):
        fig.update_layout(scene8_camera=camera, scene8_aspectmode='data')
    elif (scene == 9):
        fig.update_layout(scene9_camera=camera, scene9_aspectmode='data')
    # py.io.orca.config.port = 8999
    # pio.orca.config.use_xvfb = True
    # pio.orca.config.executable = '/usr/bin/orca'
    if(flag_plot):
        #py.offline.plot(fig)
        fig.write_html("/home/philippgfriedrich/DeepLocalShapes/plots/plot_epoch_{epoch}" + ".html")
    if(flag_screenshot):
        fig.write_image(save_dir + f'plot_epoch_{epoch}' + ".png",width=1920, height=1200,scale=2)
    #fig.layout = {}
    return fig


# function to read npz to xyz and sdf matrix. I split the npz into negative sdf values( i.e the object)
# and positive sdf( points sampled outside the mesh)
def npz2cloud(npz_filename):
    data = deep_ls.data.read_sdf_samples_into_ram(npz_filename)

    xyz_neg = data[1][:, 0:3].numpy().astype(ctypes.c_float)
    xyz_pos = data[0][:, 0:3].numpy().astype(ctypes.c_float)
    sdf_neg = data[1][:, 3].numpy().astype(ctypes.c_float)
    sdf_pos = data[0][:, 3].numpy().astype(ctypes.c_float)

    xyz_neg_browser = data[1][:, 0:4].numpy().astype(
        ctypes.c_float)  # data[0] for zero and positive values(surface and noise)
    xyz_pos_browser = data[0][:, 0:4].numpy().astype(ctypes.c_float)
    return xyz_neg_browser, xyz_pos_browser


if __name__ == "__main__":

    if not len(sys.argv) == 3:
        print("Usage: show_interior_samples.py <npz_file>")
        sys.exit(1)

    npz_filename_1 = sys.argv[1 ]
    npz_filename_2 = sys.argv[2 ]

    xyz_neg_1, xyz_pos_1 = npz2cloud(npz_filename_1)
    xyz_neg_2, xyz_pos_2 = npz2cloud(npz_filename_2)



    #print(f"positive: {len(xyz_pos_1)}")
    #print(f"negative: {len(xyz_neg_1)}")
    #print (len(xyz_pos_1) + xyz_neg_1)
    #-----------------------------------------------------------------
    #Plotlty function

    # plots one object, you can traverse positive and negative in the browser
    fig2 = go.Figure(data=[go.Scatter3d(x=xyz_neg_1[ :, 0 ], y=xyz_neg_1[ :, 1 ], z=xyz_neg_1[ :, 2 ],
                                       mode='markers',
                                       marker=dict(
                                   size=1.6,
                                   opacity=1.0,
                                   # showscale=True,
                                   colorscale="rainbow",
                                   color=xyz_neg_1[ :, 3 ].squeeze(),#xyz_neg_browser[:, 3]
                                   colorbar=dict(title='SDF_Negative-trace:0', x=0.9, y=0.4)
                               )),

                            go.Scatter3d(x=xyz_pos_1[ :, 0 ], y=xyz_pos_1[ :, 1 ], z=xyz_pos_1[ :, 2 ],
                                         mode='markers',
                                         marker=dict(
                                   size=1.6,
                                   opacity=1.0,
                                   color=xyz_pos_1[ :, 3 ],
                                   colorscale="rainbow",
                                   #showscale=True,
                                   colorbar=dict(title='SDF_Positive-trace:1', x=1, y=0.4)
                               )
                                         )
                       ],


                )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        eye=dict(x=1, y=1.1, z=-0.8)
    )

    fig2.update_layout(scene_camera=camera, scene_aspectmode='data')
    #fig.write_image("/home/mahdi/Desktop/images/tsdf/origin_sdf.png",
     # width=1920, height=1200, scale=2)
    #fig2.show()
    fig2.write_html("/home/philippgfriedrich/DeepLocalShapes/plots/3dplot2.html")


# Another example where you can plot grid of plots, I will plot only one for demonstration
    fig = make_subplots(rows=3, cols=3, specs=[ [ {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'} ],
                                                [ {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'} ],
                                                [ {'type': 'scatter3d'}, {'type': 'scatter3d'},
                                                  {'type': 'scatter3d'} ] ],
                        subplot_titles=(
                            'SDF_output_low_GT', 'SDF_output_High_GT', 'SDF_output_Pred', 'SDF_output_low_pred',
                            'SDF_output_High_pred', 'SDF_output_Pred_shell',
                            'SDF_Diff_low', 'SDF_Diff_high', 'SDF_output_diff(GT-Pred)_shell'),
                        vertical_spacing=0.05, horizontal_spacing=0.001)

     # This is supposed to be plotted during training,thats why you see these values
    epoch =1
    batch =1
    _subbatch =1
    plot_3D(xyz_neg_2[ :, 0:3 ], xyz_neg_2[ :, 3 ], _subbatch,
            [ xyz_neg_2[ :, 3 ].min(), xyz_neg_2[ :, 3 ].max(),
              f'Epoch_{epoch}: Batch_{batch} ' ], [ [ 0, 1, 0 ], [ 0.5, 1.4, -0.5 ] ],
            '/home/philippgfriedrich/DeepLocalShapes/plots/',
            flag_plot=False,
            flag_screenshot=True, fig=fig, row=2, col=1, scene=4)

    plot_3D(xyz_pos_2[ :, 0:3 ], xyz_pos_2[ :, 3 ], 2,
            [ xyz_pos_2[ :, 3 ].min(), xyz_pos_2[ :, 3 ].max(),
             f'Epoch_{epoch}: Batch_{batch} ' ], [ [ 0, 1, 0 ], [ 0.5, 1.4, -0.5 ] ],
           '/home/philippgfriedrich/DeepLocalShapes/plots/',
            flag_plot=False,
            flag_screenshot=True, fig=fig, row=2, col=1, scene=4)
