from .cal import calculator

import shutil

### 캔버스 생성 ###
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image

class Btn_img_out:
    def init(self):
        pass
        
    def forward(self, file_name, frame_out_img):
        current_preds, img_path_list = calculator().forward(file_name)
        
        fig, ax = plt.subplots(1, 1, figsize=(3.88,4.285))
        ax.imshow(Image.open(img_path_list[0]).rotate(-current_preds[0].item()))
        ax.set_title(f'기울기 : {current_preds[0]}\' (시계방향)')
        ax.grid()
        
        plt.axis('off')
        
        return  self.canv(fig, frame_out_img)
        
        
    # 캔버스 출력
    def canv(self, fig, frame_out_img):
        canvas = FigureCanvasTkAgg(fig, master=frame_out_img) 
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=3) 
        plt.close()
        
        return fig
        
    # 초기 이미지 출력
    def ferst_show(self, file_name, frame_out_img):
        fig = plt.figure(figsize=(3.88, 4.285))
        
        if file_name is None:
            file_name = r'D:\lomin\data\img_data\main_image.jpg'
        image = img.imread(file_name)
        plt.imshow(image)
        plt.axis('off')
        
        return self.canv(fig, frame_out_img)